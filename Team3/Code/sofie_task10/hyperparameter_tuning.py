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
import optuna.visualization as vis
import plotly
import plotly.graph_objects as go
import plotly.subplots as sp
import math

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
        
        # Storage for training histories across all trials
        self.trial_histories = {}
        
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
            'embed_dim': 512,
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            
            # Training
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [1]),
            
            # Class weights for handling imbalance
            # Weight for benign class (class 0), high-grade class weight fixed at 1.0
            'class_weight_benign': trial.suggest_float('class_weight_benign', 1.0, 5.0),
            'class_weight_high_grade': 1.0,  #fixed
            
            # Data sampling
            'per_slice_cap': TRAINING_CONFIG['per_slice_cap'],
            'max_slices_per_stain': TRAINING_CONFIG['max_slices_per_stain'],
            
            # Scheduler -- leave all at default
            'use_scheduler':   TRAINING_CONFIG['use_scheduler'], 
            'scheduler_type': TRAINING_CONFIG['scheduler_type'],
            'scheduler_factor': TRAINING_CONFIG['scheduler_factor'],
            'scheduler_patience': TRAINING_CONFIG['scheduler_patience'],
            
            # Early stopping #leave as default
            'early_stopping': TRAINING_CONFIG['early_stopping'], 
            'early_stopping_patience': TRAINING_CONFIG['early_stopping_patience'],
            
            # Fixed parameters
            'num_workers': TRAINING_CONFIG['num_workers'],
            'epochs': 50,  # Max epochs, will likely stop early
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
        epochs_without_improvement = 0

        # Per-trial history tracked here and stored on self for later use in _save_results
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'params': trial.params,
        }
        
        for epoch in range(trial_params['epochs']):
            # Train epoch
            train_loss = trainer.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = trainer.validate(val_loader)

            # Record epoch metrics
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['val_acc'].append(float(val_acc))
            
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
                history['pruned'] = True
                history['pruned_at_epoch'] = epoch
                self.trial_histories[trial.number] = history
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

        history['pruned'] = False
        history['best_val_loss'] = float(best_val_loss)
        self.trial_histories[trial.number] = history
        
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
            sampler=TPESampler(seed=self.seed, gamma=0.25),
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

        # ── Save all training histories as JSON ──────────────────────────────
        histories_path = os.path.join(self.study_dir, "trial_histories.json")
        with open(histories_path, 'w') as f:
            # Convert int keys to strings for valid JSON
            json.dump({str(k): v for k, v in self.trial_histories.items()}, f, indent=2)
        print(f"Training histories saved to: {histories_path}")
        
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
        
        # Create visualizations
        try:

            # ── 1. Feature importances (existing) ────────────────────────────
            fig = vis.plot_param_importances(study)
            fig.write_html(os.path.join(self.study_dir, "param_importances.html"))

            # ── 2. Trial convergence: val_loss curves for every trial ─────────
            # Separate completed vs pruned for visual clarity
            completed_trials = {
                k: v for k, v in self.trial_histories.items()
                if not v.get('pruned', False) and v.get('val_loss')
            }
            pruned_trials = {
                k: v for k, v in self.trial_histories.items()
                if v.get('pruned', False) and v.get('val_loss')
            }

            best_trial_num = study.best_trial.number
            fig_conv = go.Figure()

            # Pruned trials (faint dashed)
            for trial_num, hist in pruned_trials.items():
                epochs = list(range(1, len(hist['val_loss']) + 1))
                fig_conv.add_trace(go.Scatter(
                    x=epochs, y=hist['val_loss'],
                    mode='lines',
                    line=dict(color='rgba(180,180,180,0.35)', width=1, dash='dot'),
                    name=f"Trial {trial_num} (pruned)",
                    showlegend=False,
                    hovertemplate=f"Trial {trial_num} (pruned)<br>Epoch: %{{x}}<br>Val Loss: %{{y:.4f}}<extra></extra>"
                ))

            # Completed trials (semi-transparent)
            colorscale = plotly.colors.sample_colorscale(
                'Viridis', [i / max(len(completed_trials) - 1, 1) for i in range(len(completed_trials))]
            )
            for idx, (trial_num, hist) in enumerate(sorted(completed_trials.items())):
                epochs = list(range(1, len(hist['val_loss']) + 1))
                is_best = (trial_num == best_trial_num)
                fig_conv.add_trace(go.Scatter(
                    x=epochs, y=hist['val_loss'],
                    mode='lines',
                    line=dict(
                        color='rgba(255,80,80,1.0)' if is_best else colorscale[idx],
                        width=3 if is_best else 1.5,
                    ),
                    name=f"Trial {trial_num}" + (" ★ best" if is_best else ""),
                    hovertemplate=f"Trial {trial_num}<br>Epoch: %{{x}}<br>Val Loss: %{{y:.4f}}<extra></extra>"
                ))

            fig_conv.update_layout(
                title="Validation Loss Convergence — All Trials",
                xaxis_title="Epoch",
                yaxis_title="Validation Loss",
                template="plotly_white",
                legend=dict(orientation="v", x=1.01, y=1),
                hovermode="x unified",
            )
            fig_conv.write_html(os.path.join(self.study_dir, "trial_convergence.html"))

            # ── 3. Train vs Val loss curves per trial (multi-panel) ───────────
            all_finished = {k: v for k, v in self.trial_histories.items() if v.get('val_loss')}
            n_finished = len(all_finished)
            if n_finished > 0:
                n_cols = min(4, n_finished)
                n_rows = math.ceil(n_finished / n_cols)
                subplot_titles = [f"Trial {k}" + (" ★" if k == best_trial_num else "")
                                  for k in sorted(all_finished.keys())]
                fig_panels = sp.make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=subplot_titles,
                    shared_xaxes=False,
                    vertical_spacing=0.08,
                    horizontal_spacing=0.06,
                )
                for idx, (trial_num, hist) in enumerate(sorted(all_finished.items())):
                    row = idx // n_cols + 1
                    col = idx % n_cols + 1
                    epochs = list(range(1, len(hist['val_loss']) + 1))
                    fig_panels.add_trace(
                        go.Scatter(x=epochs, y=hist['train_loss'], mode='lines',
                                   name='train', line=dict(color='steelblue', width=1.5),
                                   showlegend=(idx == 0)),
                        row=row, col=col
                    )
                    fig_panels.add_trace(
                        go.Scatter(x=epochs, y=hist['val_loss'], mode='lines',
                                   name='val', line=dict(color='tomato', width=1.5),
                                   showlegend=(idx == 0)),
                        row=row, col=col
                    )
                fig_panels.update_layout(
                    title="Train vs Validation Loss — Per Trial",
                    template="plotly_white",
                    height=250 * n_rows + 80,
                    showlegend=True,
                )
                fig_panels.write_html(os.path.join(self.study_dir, "per_trial_loss_curves.html"))

            # ── 4. Parameter vs performance scatter plots ─────────────────────
            completed_optuna = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_optuna:
                param_names = list(completed_optuna[0].params.keys())
                trial_values = [t.value for t in completed_optuna]

                n_params = len(param_names)
                n_cols_p = min(3, n_params)
                n_rows_p = math.ceil(n_params / n_cols_p)

                fig_scatter = sp.make_subplots(
                    rows=n_rows_p, cols=n_cols_p,
                    subplot_titles=param_names,
                    vertical_spacing=0.10,
                    horizontal_spacing=0.08,
                )

                for idx, param in enumerate(param_names):
                    row = idx // n_cols_p + 1
                    col = idx % n_cols_p + 1

                    x_vals = []
                    y_vals = []
                    trial_nums = []
                    for t in completed_optuna:
                        if param in t.params:
                            raw = t.params[param]
                            # Categorical params that are None need to be stringified
                            x_vals.append(str(raw) if raw is None else raw)
                            y_vals.append(t.value)
                            trial_nums.append(t.number)

                    # Determine if x is numeric or categorical
                    numeric_vals = []
                    for v in x_vals:
                        try:
                            numeric_vals.append(float(v))
                        except (ValueError, TypeError):
                            numeric_vals = None
                            break

                    if numeric_vals is not None:
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=numeric_vals, y=y_vals,
                                mode='markers',
                                marker=dict(
                                    color=y_vals,
                                    colorscale='RdYlGn_r',
                                    showscale=(idx == 0),
                                    size=8,
                                    colorbar=dict(title="Val Loss", x=1.02) if idx == 0 else None,
                                ),
                                text=[f"Trial {n}" for n in trial_nums],
                                hovertemplate=f"{param}: %{{x}}<br>Val Loss: %{{y:.4f}}<br>%{{text}}<extra></extra>",
                                showlegend=False,
                            ),
                            row=row, col=col
                        )
                    else:
                        # Box plot grouping by categorical value
                        unique_cats = sorted(set(x_vals), key=str)
                        for cat in unique_cats:
                            cat_losses = [y for x, y in zip(x_vals, y_vals) if x == cat]
                            fig_scatter.add_trace(
                                go.Box(
                                    y=cat_losses,
                                    name=str(cat),
                                    showlegend=False,
                                    marker_color='steelblue',
                                    boxpoints='all',
                                    jitter=0.3,
                                    pointpos=0,
                                ),
                                row=row, col=col
                            )

                fig_scatter.update_layout(
                    title="Hyperparameter Values vs Validation Loss",
                    template="plotly_white",
                    height=320 * n_rows_p + 80,
                )
                fig_scatter.write_html(os.path.join(self.study_dir, "param_vs_performance.html"))

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