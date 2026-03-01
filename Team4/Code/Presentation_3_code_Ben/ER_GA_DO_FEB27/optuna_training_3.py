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

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import plotly
from tqdm import tqdm

# Import your modules
from config import DATA_PATHS, MODEL_CONFIG, TRAINING_CONFIG, SPLIT_CONFIG
from data_utils import (
    load_labels, get_all_patch_files, group_patches_by_slice,
    build_slice_to_class_map, split_by_case_stratified, build_case_dict,
    report_no_leak, summarize_case_dict
)
from models import create_model
from dataset import StainBagCasePooledFeatureDataset, case_collate_fn
from trainer import count_patches_by_class, attention_entropy
from utils import set_seed, get_device, load_data_splits

EMB_DIR = "/projects/e32998/patches_pooled4096"  # must match precompute_pooled_features.py output

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
        self.OPTUNA_params = OPTUNA_params
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
        if OPTUNA_params.get('use_scheduler', True):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=TRAINING_CONFIG.get('scheduler_factor', 0.5),
                patience=TRAINING_CONFIG.get('scheduler_patience', 3),
                min_lr=OPTUNA_params.get('scheduler_min_lr', 1e-6)
            )
        
        self.best_val_loss = float('inf')
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def _forward_one_case(
        self,
        case_data: Dict[str, Any],
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Returns (logits_with_batch, label_with_batch, attn_or_None)
        logits: (1, num_classes), label: (1,)
        """
        stain_slices = case_data["stain_slices"]
        label = case_data["label"].to(self.device)

        if return_attn:
            out = self.model(stain_slices, return_attn_weights=True)
            if not (isinstance(out, tuple) and len(out) == 2):
                raise RuntimeError("return_attn=True but model did not return (logits, attn_dict).")
            logits, attn = out
        else:
            logits = self.model(stain_slices)
            attn = None

        if logits.dim() != 1:
            raise ValueError(f"Expected model to return (num_classes,), got {tuple(logits.shape)}")

        logits = logits.unsqueeze(0)  # (1, num_classes)
        if label.dim() == 0:
            label = label.unsqueeze(0)  # (1,)

        return logits, label, attn

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()

        if len(train_loader) == 0:
            print("[WARN] train_loader is empty. Returning loss=0.0")
            self.train_losses.append(0.0)
            return 0.0

        running_loss = 0.0
        num_batches = 0
        lam = float(self.OPTUNA_params.get('entropy_lambda', 0.0))
        want_attn = lam > 0.0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            if not batch:
                continue
            case_data = batch[0]

            logits, label, attn = self._forward_one_case(case_data, return_attn=want_attn)
            base_loss = self.criterion(logits, label)

            # Entropy regularization: Loss = CE - lam * mean_entropy
            entropies = []
            if want_attn and attn is not None and "case_weights" in attn:
                entropies.append(attention_entropy(attn["case_weights"]))
                for stain, d in attn.get("stain_weights", {}).items():
                    if "slice_weights" in d:
                        entropies.append(attention_entropy(d["slice_weights"]))
                    for pw in d.get("patch_weights", []):
                        entropies.append(attention_entropy(pw))

            entropy_term = (
                torch.stack(entropies).mean()
                if entropies
                else torch.zeros(1, device=base_loss.device, dtype=base_loss.dtype).squeeze()
            )
            loss = base_loss - lam * entropy_term

            if want_attn and num_batches == 0:
                print(f"[ER DEBUG] lam={lam:.3e} base_loss={base_loss.item():.6f} "
                      f"entropy={entropy_term.item():.6f} lam*H={(lam*entropy_term).item():.6f}")

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

        if len(val_loader) == 0:
            print("[WARN] val_loader is empty. Returning loss=0.0, acc=0.0")
            self.val_losses.append(0.0)
            self.val_accuracies.append(0.0)
            return 0.0, 0.0

        total_loss = 0.0
        correct = 0
        n = 0

        with torch.inference_mode():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                if not batch:
                    continue
                case_data = batch[0]

                logits, label, _ = self._forward_one_case(case_data, return_attn=False)
                loss = self.criterion(logits, label)
                total_loss += float(loss.item())

                pred = torch.argmax(logits, dim=1)  # (1,)
                correct += int((pred == label).sum().item())
                n += 1

        avg_loss = total_loss / max(n, 1)
        accuracy = correct / max(n, 1)

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        return avg_loss, accuracy
    
    def train_with_pruning(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        epochs: int,
        pruning_warmup: int = 8
    ) -> Tuple[float, float]:
        """
        Training loop with Optuna pruning support.
        Returns: (best_val_loss, best_val_accuracy)
        """
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_acc
            
            # Optuna pruning (only after warmup period)
            if self.trial is not None and epoch >= pruning_warmup:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
        
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
    pruning_warmup: int = 8
) -> float:
    """
    Optuna objective function to minimize validation loss.
    """
    
    # ========================================================================
    # DEFINE HYPERPARAMETER SEARCH SPACE
    # ========================================================================
    OPTUNA_CONFIG = {
        # Optuna 2 found 7.31e-5 — zoom in around that
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True),

        # Optuna 2 found 7.48e-5 — zoom in around that
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),

        # Optuna 2 found 0.3447 — narrow around that
        'dropout': trial.suggest_float('dropout', 0.25, 0.45),

        # Optuna 2 found 1.85 — narrow around that
        'class_weight_benign': trial.suggest_float('class_weight_benign', 1.5, 2.2),

        # Optuna 2 found 1.47e-4 — zoom in around that
        'entropy_lambda': trial.suggest_float('entropy_lambda', 1e-5, 1e-3, log=True),

        # Optuna 2 found 1.26e-6 — narrow around that
        'scheduler_min_lr': trial.suggest_float('scheduler_min_lr', 1e-7, 5e-6, log=True),

        'use_scheduler': True
    }
    
    # ========================================================================
    # PRINT TRIAL PARAMETERS
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    print(f"Parameters:")
    print(f"  Learning rate:          {OPTUNA_CONFIG['learning_rate']:.2e}")
    print(f"  Weight decay:           {OPTUNA_CONFIG['weight_decay']:.2e}")
    print(f"  Dropout:                {OPTUNA_CONFIG['dropout']:.3f}")
    print(f"  Class weight (benign):  {OPTUNA_CONFIG['class_weight_benign']:.2f}")
    print(f"  Entropy lambda:         {OPTUNA_CONFIG['entropy_lambda']:.4f}")
    print(f"  Use scheduler:          {OPTUNA_CONFIG.get('use_scheduler', True)}")
    print(f"  Scheduler min lr:       {OPTUNA_CONFIG['scheduler_min_lr']:.2e}")
    
    # ========================================================================
    # CREATE MODEL WITH SUGGESTED HYPERPARAMETERS
    # ========================================================================
    model = create_model(
        num_classes=MODEL_CONFIG['num_classes'],
        embed_dim=MODEL_CONFIG['embed_dim'],
        dropout=OPTUNA_CONFIG['dropout']
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
            pruning_warmup=pruning_warmup
        )
        
        print(f"Trial {trial.number} completed - "
              f"Best Val Loss: {best_val_loss:.4f}, "
              f"Best Val Acc: {best_val_accuracy:.4f}")
        
        trial.set_user_attr('best_val_accuracy', best_val_accuracy)
        
        return best_val_loss
        
    except optuna.TrialPruned:
        print(f"Trial {trial.number} pruned")
        raise


# ============================================================================
# DATA PREPARATION
# ============================================================================
def prepare_data(args):
    """Prepare and split the data — mirrors main.py's prepare_data"""
    print("=" * 60)
    print("PREPARING DATA")
    print("=" * 60)

    labels = load_labels(args.labels_csv)
    print(f"Loaded {len(labels)} labels")

    all_files = get_all_patch_files(args.patches_dir)
    print(f"Found {len(all_files)} patch files")

    patches = group_patches_by_slice(all_files, args.patches_dir)
    print(f"Grouped into {len(patches)} slices")

    slice_to_class = build_slice_to_class_map(patches, labels)
    print(f"Mapped {len(slice_to_class)} slices to classes")

    slices_by_class = defaultdict(list)
    for key, label in slice_to_class.items():
        slices_by_class[label].append(key)

    print(f"Class distribution: {dict((k, len(v)) for k, v in slices_by_class.items())}")

    print("\n" + "-" * 40)
    print("SPLITTING DATA")
    print("-" * 40)

    if args.load_splits:
        print(f"Loading existing splits from: {args.load_splits}")
        splits_data = load_data_splits(args.load_splits)
        train_cases_set = set(splits_data['train_cases'])
        val_cases_set   = set(splits_data['val_cases'])

        train_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys()
                        if case_id in train_cases_set]
        val_slices   = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys()
                        if case_id in val_cases_set]

        print(f"Loaded splits - Train: {len(train_slices)}, Val: {len(val_slices)}")
    else:
        train_slices, val_slices, _ = split_by_case_stratified(
            slices_by_class, random_state=args.seed
        )
        print(f"Split sizes - Train: {len(train_slices)}, Val: {len(val_slices)}")

    train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)
    val_case_dict,   val_label_map   = build_case_dict(val_slices,   patches, slice_to_class)

    report_no_leak(train_case_dict, val_case_dict, {})

    count_patches_by_class(train_case_dict, train_label_map, "Train")
    count_patches_by_class(val_case_dict,   val_label_map,   "Validation")

    return (train_case_dict, train_label_map), (val_case_dict, val_label_map)


def create_data_loaders(train_data, val_data, per_slice_cap, max_slices_per_stain, args):
    """Create data loaders using precomputed pooled features"""
    train_case_dict, train_label_map = train_data
    val_case_dict,   val_label_map   = val_data

    train_ds = StainBagCasePooledFeatureDataset(
        train_case_dict, train_label_map,
        embeddings_dir=EMB_DIR,
        per_slice_cap=per_slice_cap,
        max_slices_per_stain=max_slices_per_stain,
        shuffle_patches=True,
    )

    val_ds = StainBagCasePooledFeatureDataset(
        val_case_dict, val_label_map,
        embeddings_dir=EMB_DIR,
        per_slice_cap=per_slice_cap,
        max_slices_per_stain=max_slices_per_stain,
        shuffle_patches=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn,
        persistent_workers=True
    )

    return train_loader, val_loader


# ============================================================================
# RESULTS SAVING AND VISUALIZATION
# ============================================================================
def save_optimization_results(study: optuna.Study, output_dir: str):
    """Save optimization results and create visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    json_path = os.path.join(output_dir, 'best_hyperparameters.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nBest hyperparameters saved to: {json_path}")
    
    df = study.trials_dataframe()
    csv_path = os.path.join(output_dir, 'all_trials.csv')
    df.to_csv(csv_path, index=False)
    print(f"All trials saved to: {csv_path}")
    
    print("\nGenerating visualizations...")
    
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(output_dir, 'optimization_history.html'))
        print("  - Optimization history saved")
    except Exception as e:
        print(f"  - Could not create optimization history: {e}")
    
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(output_dir, 'param_importances.html'))
        print("  - Parameter importances saved")
    except Exception as e:
        print(f"  - Could not create parameter importances: {e}")
    
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(output_dir, 'parallel_coordinate.html'))
        print("  - Parallel coordinate plot saved")
    except Exception as e:
        print(f"  - Could not create parallel coordinate plot: {e}")
    
    try:
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(os.path.join(output_dir, 'slice_plot.html'))
        print("  - Slice plot saved")
    except Exception as e:
        print(f"  - Could not create slice plot: {e}")
    
    try:
        importances = optuna.importance.get_param_importances(study)
        if len(importances) >= 2:
            top_params = list(importances.keys())[:2]
            fig = optuna.visualization.plot_contour(study, params=top_params)
            fig.write_html(os.path.join(output_dir, 'contour_plot.html'))
            print("  - Contour plot saved")
    except Exception as e:
        print(f"  - Could not create contour plot: {e}")
    
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
    parser = argparse.ArgumentParser(
        description='Optimize MIL model hyperparameters using Optuna'
    )
    
    parser.add_argument('--labels_csv', type=str, default=DATA_PATHS['labels_csv'])
    parser.add_argument('--patches_dir', type=str, default=DATA_PATHS['patches_dir'])
    parser.add_argument('--output_dir', type=str, default='./optuna_results')
    
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--pruning_warmup', type=int, default=3)
    parser.add_argument('--study_name', type=str, default='mil_study')
    parser.add_argument('--n_jobs', type=int, default=1)
    
    parser.add_argument('--num_workers', type=int, default=TRAINING_CONFIG['num_workers'])
    parser.add_argument('--seed', type=int, default=TRAINING_CONFIG['random_state'])
    parser.add_argument('--load_splits', type=str, default=None)
    
    parser.add_argument('--per_slice_cap', type=int, default=MODEL_CONFIG['per_slice_cap'])
    parser.add_argument('--max_slices_per_stain', type=int, default=MODEL_CONFIG['max_slices_per_stain'])
    
    parser.add_argument('--storage', type=str, default=None)
    
    return parser.parse_args()


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    if not args.storage:
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
    
    train_data, val_data = prepare_data(args)
    
    print("\n" + "=" * 60)
    print("CREATING DATA LOADERS")
    print("=" * 60)
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, args.per_slice_cap, args.max_slices_per_stain, args
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    print("\n" + "=" * 60)
    print("CREATING OPTUNA STUDY")
    print("=" * 60)
    
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(
        n_startup_trials=4,
        n_warmup_steps=args.pruning_warmup
    )
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    def objective_wrapper(trial):
        return objective(
            trial=trial,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.n_epochs,
            pruning_warmup=args.pruning_warmup
        )
    
    print("\n" + "=" * 60)
    print("STARTING OPTIMIZATION")
    print("=" * 60)
    
    study.optimize(
        objective_wrapper,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
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