#!/usr/bin/env python3
"""
Bayesian Hyperparameter Tuning for Hierarchical Attention MIL model using Optuna
"""
import os
import argparse
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from torch.utils.data import DataLoader
import json
from datetime import datetime
from collections import defaultdict

# Import your existing modules
from config import DATA_PATHS, TRAINING_CONFIG, MODEL_CONFIG
from data_utils import (
    load_labels, get_all_patch_files, group_patches_by_slice,
    build_slice_to_class_map, split_by_case_stratified, build_case_dict,
    report_no_leak, summarize_case_dict
)
from models import create_model
from dataset import StainBagCaseDataset, case_collate_fn, create_transforms
from trainer import MILTrainer, count_patches_by_class
from utils import (
    set_seed, get_device, print_data_summary, create_run_directory,
    save_data_splits, load_data_splits, print_model_summary, check_data_integrity
)


class HyperparameterTuner:
    """
    Bayesian hyperparameter tuning using Optuna
    """
    
    def __init__(self, data_splits_path=None, n_trials=50, seed=42, study_name=None):
        """
        Args:
            data_splits_path: Path to existing data splits (to ensure consistency across trials)
            n_trials: Number of optimization trials
            seed: Random seed
            study_name: Name for the Optuna study
        """
        self.data_splits_path = data_splits_path
        self.n_trials = n_trials
        self.seed = seed
        self.study_name = study_name or f"mil_hyperparam_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = get_device()
        
        # Create study directory
        self.study_dir = os.path.join("runs", self.study_name)
        os.makedirs(self.study_dir, exist_ok=True)
        
        # Load and prepare data once (reused across all trials)
        self.train_data, self.val_data, self.test_data = self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data once for all trials"""
        print("=" * 60)
        print("PREPARING DATA FOR HYPERPARAMETER TUNING")
        print("=" * 60)
        
        # Load labels
        labels = load_labels(DATA_PATHS['labels_csv'])
        print(f"Loaded {len(labels)} labels")
        
        # Get patch files
        all_files = get_all_patch_files(DATA_PATHS['patches_dir'])
        print(f"Found {len(all_files)} patch files")
        
        # Group patches by slice
        patches = group_patches_by_slice(all_files, DATA_PATHS['patches_dir'])
        print(f"Grouped into {len(patches)} slices")
        
        # Build slice to class mapping
        slice_to_class = build_slice_to_class_map(patches, labels)
        print(f"Mapped {len(slice_to_class)} slices to classes")
        
        # Group slices by class for stratified splitting
        slices_by_class = defaultdict(list)
        for key, label in slice_to_class.items():
            slices_by_class[label].append(key)
        
        print(f"Class distribution: {dict((k, len(v)) for k, v in slices_by_class.items())}")
        
        # Load or create data splits
        if self.data_splits_path and os.path.exists(self.data_splits_path):
            print(f"Loading existing splits from: {self.data_splits_path}")
            splits_data = load_data_splits(self.data_splits_path)
            train_cases_set = set(splits_data['train_cases'])
            val_cases_set = set(splits_data['val_cases'])
            test_cases_set = set(splits_data['test_cases'])
            
            train_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in train_cases_set]
            val_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in val_cases_set]
            test_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in test_cases_set]
        else:
            # Create new splits
            train_slices, val_slices, test_slices = split_by_case_stratified(
                slices_by_class, random_state=self.seed
            )
            
            # Save splits for reproducibility
            train_cases = list(set([case_id for case_id, _ in train_slices]))
            val_cases = list(set([case_id for case_id, _ in val_slices]))
            test_cases = list(set([case_id for case_id, _ in test_slices]))
            save_data_splits(train_cases, val_cases, test_cases, self.study_dir)
        
        print(f"Split sizes - Train: {len(train_slices)}, Val: {len(val_slices)}, Test: {len(test_slices)}")
        
        # Build case dictionaries
        train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)
        val_case_dict, val_label_map = build_case_dict(val_slices, patches, slice_to_class)
        test_case_dict, test_label_map = build_case_dict(test_slices, patches, slice_to_class)
        
        # Verify no data leakage
        report_no_leak(train_case_dict, val_case_dict, test_case_dict)
        
        return (train_case_dict, train_label_map), (val_case_dict, val_label_map), (test_case_dict, test_label_map)
    
    def _create_data_loaders(self, trial_params):
        """Create data loaders with trial-specific parameters"""
        train_case_dict, train_label_map = self.train_data
        val_case_dict, val_label_map = self.val_data
        test_case_dict, test_label_map = self.test_data
        
        # Create transforms
        train_transform = create_transforms(is_training=True)
        val_transform = create_transforms(is_training=False)
        
        # Create datasets with trial parameters
        train_ds = StainBagCaseDataset(
            train_case_dict, train_label_map,
            transform=train_transform,
            per_slice_cap=trial_params['per_slice_cap'],
            max_slices_per_stain=trial_params['max_slices_per_stain'],
            shuffle_patches=True,
        )
        
        val_ds = StainBagCaseDataset(
            val_case_dict, val_label_map,
            transform=val_transform,
            per_slice_cap=trial_params['per_slice_cap'],
            max_slices_per_stain=trial_params['max_slices_per_stain'],
            shuffle_patches=True,
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_ds, batch_size=trial_params['batch_size'], shuffle=True,
            num_workers=trial_params['num_workers'], pin_memory=True, 
            collate_fn=case_collate_fn, persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_ds, batch_size=trial_params['batch_size'], shuffle=False,
            num_workers=trial_params['num_workers'], pin_memory=True, 
            collate_fn=case_collate_fn, persistent_workers=True
        )
        
        return train_loader, val_loader
    
    def objective(self, trial):
        """
        Optuna objective function - defines the hyperparameter search space
        and returns the metric to optimize
        """
        # Suggest hyperparameters
        trial_params = {
            # Model architecture
            'embed_dim': trial.suggest_categorical('embed_dim', [256, 512, 768, 1024]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            
            # Training
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [1]),  # MIL typically uses batch_size=1
            
            # Class weights for handling imbalance
            # Weight for benign class (class 0), high-grade class weight fixed at 1.0
            'class_weight_benign': trial.suggest_float('class_weight_benign', 1.0, 5.0),
            'class_weight_high_grade': 1.0,  # Fixed reference point
            
            # Data sampling
            'per_slice_cap': trial.suggest_categorical('per_slice_cap', [50, 100, 150, 200]),
            'max_slices_per_stain': trial.suggest_categorical('max_slices_per_stain', [None, 5, 10, 15, 20]),
            
            # Scheduler
            'use_scheduler': trial.suggest_categorical('use_scheduler', [True, False]),
            'scheduler_type': trial.suggest_categorical('scheduler_type', ['reduce_on_plateau', 'cosine']),
            'scheduler_factor': trial.suggest_float('scheduler_factor', 0.3, 0.7),
            'scheduler_patience': trial.suggest_int('scheduler_patience', 2, 5),
            
            # Early stopping
            'early_stopping': trial.suggest_categorical('early_stopping', [True]),
            'early_stopping_patience': trial.suggest_int('early_stopping_patience', 5, 10),
            
            # Fixed parameters
            'num_workers': TRAINING_CONFIG['num_workers'],
            'epochs':10,  # Max epochs, will likely stop early
        }
        
        print(f"\n{'=' * 60}")
        print(f"Trial {trial.number}: {trial.params}")
        print(f"{'=' * 60}\n")
        
        # Set seed for reproducibility
        set_seed(self.seed + trial.number)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(trial_params)
        
        # Create model with trial parameters
        model = create_model(
            num_classes=MODEL_CONFIG['num_classes'],
            embed_dim=trial_params['embed_dim'],
            dropout=trial_params['dropout']
        )
        
        # Create checkpoint directory for this trial
        trial_checkpoint_dir = os.path.join(self.study_dir, f"trial_{trial.number}", "checkpoints")
        os.makedirs(trial_checkpoint_dir, exist_ok=True)
        
        # Create trainer
        trainer = MILTrainer(model, self.device, checkpoint_dir=trial_checkpoint_dir)
        
        # Update trainer parameters
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = trial_params['learning_rate']
            param_group['weight_decay'] = trial_params['weight_decay']
        
        # Update class weights in loss function
        class_weights = torch.tensor([
            trial_params['class_weight_benign'],
            trial_params['class_weight_high_grade']
        ]).to(self.device)
        trainer.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        # Configure scheduler
        if trial_params['use_scheduler']:
            if trial_params['scheduler_type'] == 'reduce_on_plateau':
                trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    trainer.optimizer,
                    mode='min',
                    factor=trial_params['scheduler_factor'],
                    patience=trial_params['scheduler_patience'],
                    min_lr=1e-6
                )
            elif trial_params['scheduler_type'] == 'cosine':
                trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    trainer.optimizer,
                    T_max=trial_params['epochs'],
                    eta_min=1e-6
                )
        
        # Configure early stopping
        trainer.early_stopping_patience = trial_params['early_stopping_patience']
        
        # Train with pruning callback
        best_val_loss = float('inf')
        
        for epoch in range(trial_params['epochs']):
            # Train epoch
            train_loss = trainer.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = trainer.validate(val_loader)
            
            # Update scheduler
            if trainer.scheduler:
                if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    trainer.scheduler.step(val_loss)
                else:
                    trainer.scheduler.step()
            
            # Report intermediate value for pruning
            trial.report(val_loss, epoch)
            
            # Handle pruning
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()
            
            # Track best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if trial_params['early_stopping'] and epochs_without_improvement >= trial_params['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break
            
            print(f"Epoch {epoch + 1}/{trial_params['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Clean up to free memory
        del model, trainer, train_loader, val_loader
        torch.cuda.empty_cache()
        
        # Return best validation loss (Optuna will minimize this)
        return best_val_loss
    
    def optimize(self):
        """Run the Bayesian optimization"""
        print(f"\n{'=' * 80}")
        print(f"STARTING BAYESIAN HYPERPARAMETER OPTIMIZATION")
        print(f"Study name: {self.study_name}")
        print(f"Number of trials: {self.n_trials}")
        print(f"{'=' * 80}\n")
        
        # Create Optuna study
        study = optuna.create_study(
            study_name=self.study_name,
            direction='minimize',  # Minimize validation loss
            sampler=TPESampler(seed=self.seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Print results
        print(f"\n{'=' * 80}")
        print("OPTIMIZATION COMPLETED")
        print(f"{'=' * 80}\n")
        
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation loss: {study.best_value:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Save results
        self._save_results(study)
        
        return study
    
    def _save_results(self, study):
        """Save optimization results"""
        # Save best parameters as JSON
        best_params_path = os.path.join(self.study_dir, "best_hyperparameters.json")
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        print(f"\nBest hyperparameters saved to: {best_params_path}")
        
        # Save all trials as CSV
        trials_df = study.trials_dataframe()
        trials_csv_path = os.path.join(self.study_dir, "all_trials.csv")
        trials_df.to_csv(trials_csv_path, index=False)
        print(f"All trials saved to: {trials_csv_path}")
        
        # Save study summary
        summary_path = os.path.join(self.study_dir, "optimization_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Hyperparameter Optimization Summary\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"Study name: {study.study_name}\n")
            f.write(f"Number of trials: {len(study.trials)}\n")
            f.write(f"Best trial: {study.best_trial.number}\n")
            f.write(f"Best validation loss: {study.best_value:.4f}\n\n")
            f.write(f"Best hyperparameters:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n\nTop 5 trials:\n")
            for i, trial in enumerate(sorted(study.trials, key=lambda t: t.value)[:5]):
                f.write(f"\n{i+1}. Trial {trial.number}: {trial.value:.4f}\n")
                for key, value in trial.params.items():
                    f.write(f"   {key}: {value}\n")
        
        print(f"Optimization summary saved to: {summary_path}")
        
        # Create visualization if optuna.visualization is available
        try:
            import optuna.visualization as vis
            import plotly
            
            # Optimization history
            fig = vis.plot_optimization_history(study)
            fig.write_html(os.path.join(self.study_dir, "optimization_history.html"))
            
            # Parameter importance
            fig = vis.plot_param_importances(study)
            fig.write_html(os.path.join(self.study_dir, "param_importances.html"))
            
            # Parallel coordinate plot
            fig = vis.plot_parallel_coordinate(study)
            fig.write_html(os.path.join(self.study_dir, "parallel_coordinate.html"))
            
            print(f"Visualizations saved to: {self.study_dir}")
        except ImportError:
            print("Install plotly for visualizations: pip install plotly")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bayesian Hyperparameter Tuning for MIL model')
    
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--study_name', type=str, default=None,
                       help='Name for the study (auto-generated if not provided)')
    parser.add_argument('--data_splits', type=str, default=None,
                       help='Path to existing data_splits.npz file')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create tuner
    tuner = HyperparameterTuner(
        data_splits_path=args.data_splits,
        n_trials=args.n_trials,
        seed=args.seed,
        study_name=args.study_name
    )
    
    # Run optimization
    study = tuner.optimize()
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 80)
    print("\nTo train with best hyperparameters, update your config.py or use:")
    print(f"  --embed_dim {study.best_params['embed_dim']}")
    print(f"  --lr {study.best_params['learning_rate']}")
    print(f"  --per_slice_cap {study.best_params['per_slice_cap']}")
    print("  ... etc")


if __name__ == "__main__":
    main()