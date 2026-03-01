#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for Hierarchical Attention MIL model
"""
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Dict, Any, Tuple
from datetime import datetime
import json
from tqdm import tqdm

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import plotly

# Import your modules
from config import DATA_PATHS, MODEL_CONFIG, TRAINING_CONFIG, SPLIT_CONFIG
from data_utils import (
    load_labels, get_all_patch_files, group_patches_by_slice,
    build_slice_to_class_map, split_by_case_stratified, build_case_dict,
    report_no_leak
)
from models import create_model
from dataset import StainBagCasePooledFeatureDataset, case_collate_fn
from utils import set_seed, get_device, load_data_splits
from trainer import attention_entropy

EMB_DIR = "/projects/e32998/patches_pooled4096" 

# ============================================================================
# OPTUNA TRAINER CLASS
# ============================================================================
class OptunaTrainer:
    """
    Trainer class for Optuna optimization with pruning support
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        device: str,
        OPTUNA_params: Dict[str, Any],
        trial: optuna.Trial = None
    ):
        self.model = model
        self.device = device
        self.trial = trial
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=OPTUNA_params['learning_rate'],
            weight_decay=OPTUNA_params['weight_decay']
        )
        
        # Initialize criterion with class weights
        class_weights_tensor = torch.tensor([
            OPTUNA_params['class_weight_benign'], 
            1.0
        ]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Setup scheduler if enabled
        self.scheduler = None
        if TRAINING_CONFIG.get('use_scheduler', True):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=TRAINING_CONFIG.get('scheduler_factor', 0.2139),
                patience=TRAINING_CONFIG.get('scheduler_patience', 4),
                min_lr=TRAINING_CONFIG.get('scheduler_min_lr', 1e-6)
            )
        
        self.best_val_loss = float('inf')
        self._printed_entropy_debug = False
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def _forward_one_case(self, case_data: Dict[str, Any], return_attn_weights: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits_with_batch, label_with_batch)
        logits_with_batch: shape (1, num_classes)
        label_with_batch:  shape (1,)
        """
        stain_slices = case_data["stain_slices"]
        label = case_data["label"].to(self.device)
        
        if return_attn_weights:
            logits, attn_weights = self.model(stain_slices, return_attn_weights=True)
        else:
            logits = self.model(stain_slices)  # (num_classes,)
            attn_weights = None


        if logits.dim() != 1:
            raise ValueError(f"Expected model to return (num_classes,), got {tuple(logits.shape)}")

        logits = logits.unsqueeze(0)  # (1, num_classes)

        if label.dim() == 0:
            label = label.unsqueeze(0)  # (1,)

        if return_attn_weights:
            return logits, label, attn_weights
        return logits, label
    
    def train_epoch(self, train_loader: DataLoader, entropy_lambda: float) -> float:
        """
        Train for one epoch.
        Returns average training loss.
        """
        self.model.train()

        if len(train_loader) == 0:
            print("[WARN] train_loader is empty. Returning loss=0.0")
            self.train_losses.append(0.0)
            return 0.0

        running_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            if not batch:
                continue
            case_data = batch[0]

            lam = entropy_lambda if entropy_lambda is not None else 0.0

            # Foward pass
            if lam > 0.0:
                logits, label, attn = self._forward_one_case(case_data, return_attn_weights=True)
            else:
                logits, label = self._forward_one_case(case_data)
                attn = None
            
            base_loss = self.criterion(logits, label)
            
            # Entropy regularization averaged across all attention layers
            entropies = []
            
            if lam > 0.0:
                if attn is None:
                    raise RuntimeError("Entropy reg enabled but attention weights were not returned correctly.")
                
                # Only calculate entropy if the case actually has valid weights
                if "case_weights" in attn:
                    entropies.append(attention_entropy(attn["case_weights"]))
            
                    for stain, d in attn.get("stain_weights", {}).items():
                        if "slice_weights" in d:
                            entropies.append(attention_entropy(d["slice_weights"]))
                        for pw in d.get("patch_weights", []):
                            entropies.append(attention_entropy(pw))
            
            if len(entropies) > 0:
                entropy_term = torch.stack(entropies).mean()
            else:
                entropy_term = torch.tensor(0.0, device=self.device)
            
            # Loss = CE - lambda * H(w)
            loss = base_loss - lam * entropy_term
            
            #### DEBUGGING #####
            if (lam > 0.0) and (not self._printed_entropy_debug):
                penalty_mag = (lam * entropy_term).detach().item()
                base = base_loss.detach().item()
                ent = entropy_term.detach().item()

                print("\n[EntropyReg DEBUG — first batch only]")
                print(f"  entropy_lambda (lam): {lam:.3e}")
                print(f"  base_loss (CE):        {base:.6f}")
                print(f"  entropy_term (mean H): {ent:.6f}")
                print(f"  lam * entropy_term:    {penalty_mag:.6f}")
                if base != 0:
                    print(f"  (lam*H)/CE ratio:      {penalty_mag / base:.6f}")

                if not torch.isfinite(base_loss):
                    raise RuntimeError("base_loss is not finite.")
                if not torch.isfinite(entropy_term):
                    raise RuntimeError("entropy_term is not finite.")
                if not torch.isfinite(loss):
                    raise RuntimeError("total loss is not finite.")

                self._printed_entropy_debug = True
            #### END DEBUGGING ####
            
            if not loss.requires_grad:
                continue

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct_total = 0
        sample_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                case_data = batch[0]
                stain_slices = case_data["stain_slices"]
                label = case_data["label"].to(self.device)
                
                outputs = self.model(stain_slices)
                outputs = outputs.unsqueeze(0)
                
                if label.dim() == 0:
                    label = label.unsqueeze(0)
                
                loss = self.criterion(outputs, label)
                val_loss += loss.item()
                
                pred = torch.argmax(outputs, dim=1)
                correct_total += (pred == label).sum().item()
                sample_total += 1
        
        avg_loss = val_loss / max(sample_total, 1)
        accuracy = correct_total / max(sample_total, 1)
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def train_with_pruning(
            self, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            epochs: int,
            pruning_warmup: int = 6,
            entropy_lambda: float = None
        ) -> Tuple[float, float]:
            """
            Training loop with Optuna pruning support and verbose progress printing
            Returns: (best_val_loss, best_val_accuracy)
            """
            best_val_loss = float('inf')
            best_val_accuracy = 0.0
            best_epoch = 0

            # Early stopping setup
            use_early_stopping = TRAINING_CONFIG.get('early_stopping', False)
            es_patience = TRAINING_CONFIG.get('early_stopping_patience', 8)
            es_min_delta = TRAINING_CONFIG.get('early_stopping_min_delta', 0.001)
            es_min_epochs = TRAINING_CONFIG.get('early_stopping_min_epochs', 10)
            epochs_without_improvement = 0

            print(f"\n{'─'*60}")
            print(f"Starting Training - {epochs} epochs")
            print(f"{'─'*60}")
            
            for epoch in range(epochs):
                # Train
                train_loss = self.train_epoch(train_loader, entropy_lambda=entropy_lambda)
                
                # Validate
                val_loss, val_acc = self.validate(val_loader)
                
                # Update learning rate scheduler
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler:
                    old_lr = current_lr
                    self.scheduler.step(val_loss)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    lr_changed = (old_lr != current_lr)
                else:
                    lr_changed = False
                
                # Track best metrics
                improved = False
                if val_loss < best_val_loss - es_min_delta:
                    best_val_loss = val_loss
                    best_val_accuracy = val_acc
                    best_epoch = epoch
                    improved = True
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                # Print epoch results
                status = "✓ BEST" if improved else ""
                lr_info = f" [LR: {current_lr:.2e}]" if lr_changed else ""
                
                print(f"Epoch {epoch + 1:3d}/{epochs} │ "
                    f"Train Loss: {train_loss:.4f} │ "
                    f"Val Loss: {val_loss:.4f} │ "
                    f"Val Acc: {val_acc:.4f} │ "
                    f"LR: {current_lr:.2e}{lr_info} {status}")
                
                # Optuna pruning (only after warmup period)
                if self.trial is not None and epoch >= pruning_warmup:
                    # Report intermediate value
                    self.trial.report(val_loss, epoch)
                    
                    # Check if trial should be pruned
                    if self.trial.should_prune():
                        print(f"\n{'!'*60}")
                        print(f"  TRIAL PRUNED at epoch {epoch + 1}")
                        print(f"   Reason: Performance not improving compared to other trials")
                        print(f"   Best val loss so far: {best_val_loss:.4f} (epoch {best_epoch + 1})")
                        print(f"{'!'*60}\n")
                        raise optuna.TrialPruned()
                        
                # Early stopping check
                if use_early_stopping and epoch >= es_min_epochs - 1:
                    if epochs_without_improvement >= es_patience:
                        print(f"\n{'─'*60}")
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        print(f"  No improvement for {epochs_without_improvement} epochs")
                        print(f"  Best val loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
                        print(f"{'─'*60}\n")
                        break
            
            # Training completed
            print(f"\n{'─'*60}")
            print(f"Training Completed")
            print(f"  Best Val Loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
            print(f"  Best Val Accuracy: {best_val_accuracy:.4f}")
            print(f"{'─'*60}\n")
            
            return best_val_loss, best_val_accuracy


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================
def objective(
    trial: optuna.Trial, 
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    pruning_warmup: int = 6 #increase to 6
) -> float:
    """
    Optuna objective function to minimize validation loss
    
    Args:
        trial: Optuna trial object
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of epochs to train
        pruning_warmup: Number of epochs before pruning can start
    
    Returns:
        Best validation loss achieved during training
    """
    
    # ========================================================================
    # DEFINE HYPERPARAMETER SEARCH SPACE
    # ========================================================================
    OPTUNA_CONFIG = {}
    
    # Suggest hyperparameters
    OPTUNA_CONFIG['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-4, log=True) #tune initial learning rate on scheduler
    OPTUNA_CONFIG['weight_decay'] = trial.suggest_float('weight_decay', 5e-5, 5e-4, log=True) #tune weight decay
    OPTUNA_CONFIG['patch_proj_dropout'] = trial.suggest_float('patch_proj_dropout', 0.2, 0.5) #tune patch projector dropout
    OPTUNA_CONFIG['classifier_dropout'] = trial.suggest_float('classifier_dropout', 0.2, 0.5) #tune classifier dropout
    OPTUNA_CONFIG['class_weight_benign'] = trial.suggest_float('class_weight_benign', 1.5, 3.5) #tune class weight for benign class (keep malignant fixed at 1.0)
    OPTUNA_CONFIG['entropy_lambda'] = trial.suggest_float('entropy_lambda', 1e-4, 1e-2, log=True) #tune entropy regularization lambda
    
    # ========================================================================
    # PRINT TRIAL PARAMETERS
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    print(f"Parameters:")
    print(f"  Learning rate: {OPTUNA_CONFIG['learning_rate']:.2e}")
    print(f"  Weight decay: {OPTUNA_CONFIG['weight_decay']:.2e}")
    print(f"  Patch Projector Dropout: {OPTUNA_CONFIG['patch_proj_dropout']:.3f}")
    print(f"  Classifier Dropout: {OPTUNA_CONFIG['classifier_dropout']:.3f}")
    print(f"  Class weight (benign): {OPTUNA_CONFIG['class_weight_benign']:.2f}")
    print(f"  Entropy lambda: {OPTUNA_CONFIG['entropy_lambda']:.2e}")
    
    # ========================================================================
    # CREATE MODEL WITH SUGGESTED HYPERPARAMETERS
    # ========================================================================
    # Use embed_dim and attention_hidden_dim from MODEL_CONFIG (not optimized)
    model = create_model(
        num_classes=MODEL_CONFIG['num_classes'],
        embed_dim=MODEL_CONFIG['embed_dim'],
        patch_proj_dropout=OPTUNA_CONFIG['patch_proj_dropout'],
        classifier_dropout=OPTUNA_CONFIG['classifier_dropout']
    )
    
    # ========================================================================
    # CREATE TRAINER
    # ========================================================================
    trainer = OptunaTrainer(
        model=model,
        device=device,
        OPTUNA_params=OPTUNA_CONFIG,
        trial=trial
    )
    
    # ========================================================================
    # TRAIN WITH PRUNING
    # ========================================================================
    try:
        best_val_loss, best_val_accuracy = trainer.train_with_pruning(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            pruning_warmup=pruning_warmup, 
            entropy_lambda=OPTUNA_CONFIG['entropy_lambda']
        )
        
        print(f"Trial {trial.number} completed - "
              f"Best Val Loss: {best_val_loss:.4f}, "
              f"Best Val Acc: {best_val_accuracy:.4f}")
        
        # Store best accuracy as user attribute for later analysis
        trial.set_user_attr('best_val_accuracy', best_val_accuracy)
        
        return best_val_loss
        
    except optuna.TrialPruned:
        print(f"Trial {trial.number} pruned")
        raise


# ============================================================================
# DATA PREPARATION
# ============================================================================
def prepare_data(args):
    """Prepare and split the data"""
    print("=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    
    # Load labels
    labels = load_labels(args.labels_csv)
    print(f"Loaded {len(labels)} labels")
    
    # Get patch files
    all_files = get_all_patch_files(args.patches_dir)
    print(f"Found {len(all_files)} patch files")
    
    # Group patches by slice
    patches = group_patches_by_slice(all_files, args.patches_dir)
    print(f"Grouped into {len(patches)} slices")
    
    # Build slice to class mapping
    slice_to_class = build_slice_to_class_map(patches, labels)
    print(f"Mapped {len(slice_to_class)} slices to classes")
    
    # Group slices by class for stratified splitting
    slices_by_class = defaultdict(list)
    for key, label in slice_to_class.items():
        slices_by_class[label].append(key)
    
    print(f"Class distribution: {dict((k, len(v)) for k, v in slices_by_class.items())}")
    
    print("\n" + "-" * 40)
    print("SPLITTING DATA")
    print("-" * 40)
    
    if args.load_splits:
        # Load existing splits
        print(f"Loading existing splits from: {args.load_splits}")
        splits_data = load_data_splits(args.load_splits)
        train_cases_set = set(splits_data['train_cases'])
        val_cases_set = set(splits_data['val_cases'])
        
        train_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() 
                        if case_id in train_cases_set]
        val_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() 
                      if case_id in val_cases_set]
        
        print(f"Loaded splits - Train: {len(train_slices)}, Val: {len(val_slices)}")
    else:
        # Split data by case (stratified) - only need train and val for optimization
        train_slices, val_slices, _ = split_by_case_stratified(
            slices_by_class, random_state=args.seed
        )
        
        print(f"Split sizes - Train: {len(train_slices)}, Val: {len(val_slices)}")
    
    # Build case dictionaries
    train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)
    val_case_dict, val_label_map = build_case_dict(val_slices, patches, slice_to_class)
    
    # Check for data leakage
    report_no_leak(train_case_dict, val_case_dict, {})
    
    return (train_case_dict, train_label_map), (val_case_dict, val_label_map)


def create_data_loaders(train_data, val_data, args):
    """Create data loaders (precomputed pooled features)"""
    print("\n" + "=" * 60)
    print("CREATING DATA LOADERS (POOLED FEATURES)")
    print("=" * 60)

    train_case_dict, train_label_map = train_data
    val_case_dict, val_label_map = val_data

    train_ds = StainBagCasePooledFeatureDataset(
        train_case_dict, train_label_map,
        embeddings_dir=EMB_DIR,
        per_slice_cap=args.per_slice_cap,
        max_slices_per_stain=args.max_slices_per_stain,
        shuffle_patches=True,
    )

    val_ds = StainBagCasePooledFeatureDataset(
        val_case_dict, val_label_map,
        embeddings_dir=EMB_DIR,
        per_slice_cap=args.per_slice_cap,
        max_slices_per_stain=args.max_slices_per_stain,
        shuffle_patches=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn,
        persistent_workers=True,
    )

    print(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}")
    return train_loader, val_loader


# ============================================================================
# RESULTS SAVING AND VISUALIZATION
# ============================================================================
def save_optimization_results(study: optuna.Study, output_dir: str):
    """Save optimization results and create visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best parameters
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial
    
    results = {
        'best_trial_number': best_trial.number,
        'best_val_loss': best_value,
        'best_val_accuracy': best_trial.user_attrs.get('best_val_accuracy', None),
        'best_params': best_params,
        'n_trials': len(study.trials),
        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save as JSON
    json_path = os.path.join(output_dir, 'best_hyperparameters.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nBest hyperparameters saved to: {json_path}")
    
    # Save all trials to CSV
    df = study.trials_dataframe()
    csv_path = os.path.join(output_dir, 'all_trials.csv')
    df.to_csv(csv_path, index=False)
    print(f"All trials saved to: {csv_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Optimization history
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(output_dir, 'optimization_history.html'))
        print("  - Optimization history saved")
    except Exception as e:
        print(f"  - Could not create optimization history: {e}")
    
    # 2. Parameter importance
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(output_dir, 'param_importances.html'))
        print("  - Parameter importances saved")
    except Exception as e:
        print(f"  - Could not create parameter importances: {e}")
    
    # 3. Parallel coordinate plot
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(output_dir, 'parallel_coordinate.html'))
        print("  - Parallel coordinate plot saved")
    except Exception as e:
        print(f"  - Could not create parallel coordinate plot: {e}")
    
    # 4. Slice plot
    try:
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(os.path.join(output_dir, 'slice_plot.html'))
        print("  - Slice plot saved")
    except Exception as e:
        print(f"  - Could not create slice plot: {e}")
    
    # 5. Contour plot for top 2 important parameters
    try:
        importances = optuna.importance.get_param_importances(study)
        if len(importances) >= 2:
            top_params = list(importances.keys())[:2]
            fig = optuna.visualization.plot_contour(study, params=top_params)
            fig.write_html(os.path.join(output_dir, 'contour_plot.html'))
            print("  - Contour plot saved")
    except Exception as e:
        print(f"  - Could not create contour plot: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best validation loss: {best_value:.4f}")
    if 'best_val_accuracy' in best_trial.user_attrs:
        print(f"Best validation accuracy: {best_trial.user_attrs['best_val_accuracy']:.4f}")
    print(f"\nBest hyperparameters:")
    for param, value in best_params.items():
        if isinstance(value, float):
            if value < 0.01:
                print(f"  {param}: {value:.2e}")
            else:
                print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Optimize MIL model hyperparameters using Optuna'
    )
    
    # Data arguments
    parser.add_argument('--labels_csv', type=str, default=DATA_PATHS['labels_csv'],
                       help='Path to labels CSV file')
    parser.add_argument('--patches_dir', type=str, default=DATA_PATHS['patches_dir'],
                       help='Path to patches directory')
    parser.add_argument('--output_dir', type=str, default='./optuna_results',
                       help='Directory to save optimization results')
    
    # Optuna arguments
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials')
    parser.add_argument('--n_epochs', type=int, default=15,
                       help='Number of epochs per trial (reduced for faster tuning)')
    parser.add_argument('--pruning_warmup', type=int, default=6,#increase to 6
                       help='Number of epochs before pruning starts')
    parser.add_argument('--study_name', type=str, default='mil_study',
                       help='Name of the Optuna study')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel jobs (set >1 for multi-GPU)')
    
    # Data loader arguments
    parser.add_argument('--num_workers', type=int, default=TRAINING_CONFIG['num_workers'],
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=TRAINING_CONFIG['random_state'],
                       help='Random seed')
    parser.add_argument('--load_splits', type=str, default=None,
                       help='Path to data_splits.npz file to load existing splits')
    parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'],
                       help='Batch size for data loaders')
    
    # Dataset configuration (NOT optimized by Optuna)
    parser.add_argument('--per_slice_cap', type=int, default=MODEL_CONFIG['per_slice_cap'],
                       help='Maximum patches per slice')
    parser.add_argument('--max_slices_per_stain', type=int, default=MODEL_CONFIG['max_slices_per_stain'],
                       help='Maximum slices per stain (None for unlimited)')
    
    # Storage
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL for study storage (optional)')
    
    return parser.parse_args()


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main optimization function"""
    args = parse_args()
    
    # Set up
    set_seed(args.seed)
    device = get_device()
    
    # Create output directory
    if not args.storage:
        # Add timestamp to prevent overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{args.output_dir}_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION FOR MIL")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Epochs per trial: {args.n_epochs}")
    print(f"Pruning warmup: {args.pruning_warmup} epochs")
    print(f"Per slice cap: {args.per_slice_cap}")
    print(f"Max slices per stain: {args.max_slices_per_stain}")
    print(f"Output directory: {args.output_dir}")
    
    # Prepare data (only once)
    train_data, val_data = prepare_data(args)
    
    # Create data loaders once (since per_slice_cap is fixed)
    print("\n" + "=" * 60)
    print("CREATING DATA LOADERS")
    print("=" * 60)
    train_loader, val_loader = create_data_loaders(train_data, val_data, args)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create Optuna study
    print("\n" + "=" * 60)
    print("CREATING OPTUNA STUDY")
    print("=" * 60)
    
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(
        n_startup_trials=4,  # Don't prune first 5 trials
        n_warmup_steps=args.pruning_warmup  # Don't prune before warmup epochs
    )
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',  # Minimize validation loss
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True  # Resume if study exists
    )
    
    # Define objective wrapper with fixed data loaders
    def objective_wrapper(trial):
        return objective(
            trial=trial,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.n_epochs,
            pruning_warmup=args.pruning_warmup
        )
    
    # Run optimization
    print("\n" + "=" * 60)
    print("STARTING OPTIMIZATION")
    print("=" * 60)
    
    study.optimize(
        objective_wrapper,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    # Save results and create visualizations
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    save_optimization_results(study, args.output_dir)
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print(f"View visualizations by opening the HTML files in: {args.output_dir}")


if __name__ == "__main__":
    main()








