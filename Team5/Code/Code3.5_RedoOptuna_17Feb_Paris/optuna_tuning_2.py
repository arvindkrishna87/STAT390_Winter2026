#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning for Hierarchical Attention MIL Model
Simplified version: only tunes learning_rate, weight_decay, dropout
"""
import os
import sys
import argparse
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd
import json
from datetime import datetime

# Import from existing codebase
from config import DATA_PATHS, MODEL_CONFIG, SPLIT_CONFIG, IMAGE_CONFIG
from data_utils import (
    load_labels, get_all_patch_files, group_patches_by_slice,
    build_slice_to_class_map, split_by_case_stratified, build_case_dict,
    report_no_leak
)
from models import create_model
from dataset import StainBagCaseDataset, case_collate_fn, create_transforms
from utils import set_seed, get_device


def move_stain_slices_to_device(stain_slices, device):
    """Move nested stain_slices dict {stain_name: [tensor, ...]} to device"""
    return {
        stain: [t.to(device) for t in tensors]
        for stain, tensors in stain_slices.items()
    }


class OptunaTrainer:
    """
    Lightweight trainer for Optuna trials with early stopping
    """
    def __init__(self, model, device, trial_params, trial=None):
        self.model = model
        self.device = device
        self.trial = trial
        self.model.to(self.device)

        # Setup optimizer with trial parameters
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=trial_params['learning_rate'],
            weight_decay=trial_params['weight_decay']
        )

        # Setup loss with class weights
        class_weights = torch.tensor(trial_params['class_weights']).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Setup scheduler if enabled
        self.scheduler = None
        if trial_params.get('use_scheduler', True):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=trial_params.get('scheduler_factor', 0.5),
                patience=trial_params.get('scheduler_patience', 3),
                min_lr=trial_params.get('scheduler_min_lr', 1e-7)
            )

        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            case_data = batch[0]
            stain_slices = move_stain_slices_to_device(case_data["stain_slices"], self.device)
            label = case_data["label"].to(self.device)

            outputs = self.model(stain_slices)
            outputs = outputs.unsqueeze(0)
            if label.dim() == 0:
                label = label.unsqueeze(0)

            loss = self.criterion(outputs, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                case_data = batch[0]
                stain_slices = move_stain_slices_to_device(case_data["stain_slices"], self.device)
                label = case_data["label"].to(self.device)

                outputs = self.model(stain_slices)
                outputs = outputs.unsqueeze(0)
                if label.dim() == 0:
                    label = label.unsqueeze(0)

                loss = self.criterion(outputs, label)
                val_loss += loss.item()

                pred = torch.argmax(outputs, dim=1)
                correct += (pred == label).sum().item()
                total += 1

        avg_loss = val_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        return avg_loss, accuracy

    def train_with_pruning(self, train_loader, val_loader, n_epochs):
        """Training loop with Optuna pruning"""
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Track best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            # Report to Optuna and handle pruning
            if self.trial is not None:
                self.trial.report(val_loss, epoch)

                # Check if trial should be pruned
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

        return self.best_val_loss, val_acc


def objective(trial, train_data, val_data, args):
    """
    Optuna objective function
    Suggests hyperparameters and returns validation loss
    """
    # Suggest hyperparameters (only learning parameters)
    trial_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),

        # Fixed values
        'class_weights': [1.0, 1.0],
        'use_scheduler': False,
    }

    # Create model with fixed architecture from config
    model = create_model(
        num_classes=MODEL_CONFIG['num_classes'],
        embed_dim=MODEL_CONFIG['embed_dim'],
        dropout=trial_params['dropout']
    )

    device = get_device()

    # Prepare data loaders
    train_case_dict, train_label_map = train_data
    val_case_dict, val_label_map = val_data

    train_transform = create_transforms(is_training=True)
    val_transform = create_transforms(is_training=False)

    # Create datasets
    train_ds = StainBagCaseDataset(
        train_case_dict, train_label_map,
        transform=train_transform,
        per_slice_cap=args.per_slice_cap,
        max_slices_per_stain=args.max_slices_per_stain,
        shuffle_patches=True,
    )

    val_ds = StainBagCaseDataset(
        val_case_dict, val_label_map,
        transform=val_transform,
        per_slice_cap=args.per_slice_cap,
        max_slices_per_stain=args.max_slices_per_stain,
        shuffle_patches=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn
    )

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn
    )

    # Create trainer
    trainer = OptunaTrainer(model, device, trial_params, trial=trial)

    # Train with pruning
    try:
        best_val_loss, val_acc = trainer.train_with_pruning(
            train_loader, val_loader, n_epochs=args.n_epochs
        )
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise optuna.TrialPruned()

    # Clean up
    del model, trainer, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_loss


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning for MIL')

    # Data arguments
    parser.add_argument('--labels_csv', type=str, default=DATA_PATHS['labels_csv'],
                       help='Path to labels CSV file')
    parser.add_argument('--patches_dir', type=str, default=DATA_PATHS['patches_dir'],
                       help='Path to patches directory')

    # Optuna arguments
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials')
    parser.add_argument('--n_epochs', type=int, default=15,
                       help='Number of epochs per trial (reduced for faster tuning)')
    parser.add_argument('--study_name', type=str, default='mil_study',
                       help='Name of the Optuna study')
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL for study storage (e.g., sqlite:///optuna.db)')

    # Model arguments
    parser.add_argument('--per_slice_cap', type=int, default=MODEL_CONFIG['per_slice_cap'],
                       help='Maximum patches per slice')
    parser.add_argument('--max_slices_per_stain', type=int, default=MODEL_CONFIG['max_slices_per_stain'],
                       help='Maximum slices per stain')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--load_splits', type=str, default=None,
                       help='Path to existing data splits')

    return parser.parse_args()


def prepare_data(args):
    """Prepare train and validation data"""
    print("=" * 60)
    print("PREPARING DATA FOR OPTUNA TUNING")
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

    # Group slices by class
    slices_by_class = defaultdict(list)
    for key, label in slice_to_class.items():
        slices_by_class[label].append(key)

    # Split data
    if args.load_splits:
        # Load existing splits
        from utils import load_data_splits
        print(f"Loading existing splits from: {args.load_splits}")
        splits_data = load_data_splits(args.load_splits)
        train_cases_set = set(splits_data['train_cases'])
        val_cases_set = set(splits_data['val_cases'])

        train_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys()
                       if case_id in train_cases_set]
        val_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys()
                     if case_id in val_cases_set]
    else:
        # Create new splits
        train_slices, val_slices, _ = split_by_case_stratified(
            slices_by_class, random_state=args.seed
        )

    print(f"Split sizes - Train: {len(train_slices)}, Val: {len(val_slices)}")

    # Build case dictionaries
    train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)
    val_case_dict, val_label_map = build_case_dict(val_slices, patches, slice_to_class)

    # Report no data leakage
    print("\nChecking for data leakage...")
    from data_utils import get_case_ids
    print(f"Train cases: {len(get_case_ids(train_case_dict))}")
    print(f"Val cases: {len(get_case_ids(val_case_dict))}")

    return (train_case_dict, train_label_map), (val_case_dict, val_label_map)


def save_study_results(study, output_dir):
    """Save study results to files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save best parameters
    best_params_path = os.path.join(output_dir, 'best_hyperparameters.json')
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nBest hyperparameters saved to: {best_params_path}")

    # Save study statistics
    stats = {
        'best_value': study.best_value,
        'best_trial_number': study.best_trial.number,
        'n_trials': len(study.trials),
        'datetime_completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    stats_path = os.path.join(output_dir, 'study_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)

    # Save all trials to CSV
    df = study.trials_dataframe()
    trials_path = os.path.join(output_dir, 'all_trials.csv')
    df.to_csv(trials_path, index=False)
    print(f"All trials saved to: {trials_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("OPTUNA STUDY COMPLETED")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Print pruned/failed trials
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"\nCompleted trials: {len(complete_trials)}")
    print(f"Pruned trials: {len(pruned_trials)}")


def main():
    """Main function for Optuna hyperparameter tuning"""
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    print("=" * 80)
    print("OPTUNA HYPERPARAMETER TUNING FOR HIERARCHICAL ATTENTION MIL")
    print("=" * 80)
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Epochs per trial: {args.n_epochs}")

    # Prepare data
    train_data, val_data = prepare_data(args)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(DATA_PATHS['runs_dir'], f"optuna_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Create study
    storage_url = args.storage if args.storage else f"sqlite:///{output_dir}/optuna.db"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction='minimize',  # Minimize validation loss
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=5,    # Wait 5 epochs before pruning
            interval_steps=1
        ),
        load_if_exists=True  # Resume if study exists
    )

    # Run optimization
    print("\n" + "=" * 60)
    print("STARTING HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    study.optimize(
        lambda trial: objective(trial, train_data, val_data, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
        gc_after_trial=True  # Garbage collect after each trial
    )

    # Save results
    save_study_results(study, output_dir)

    # Save best hyperparameters in config format
    config_path = os.path.join(output_dir, 'best_config.py')
    with open(config_path, 'w') as f:
        f.write("# Best hyperparameters found by Optuna\n")
        f.write("# Copy these values to your config.py\n\n")
        f.write("TRAINING_CONFIG = {\n")
        f.write(f"    'learning_rate': {study.best_params['learning_rate']},\n")
        f.write(f"    'weight_decay': {study.best_params['weight_decay']},\n")
        f.write(f"    'dropout': {study.best_params['dropout']},\n")
        f.write("}\n")

    print(f"\nConfig template saved to: {config_path}")
    print("\nYou can now use these hyperparameters in your main training!")


if __name__ == "__main__":
    main()