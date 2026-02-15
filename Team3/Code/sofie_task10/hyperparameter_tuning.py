#import necessary libraries
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
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

#import existing modules from other files
from config import DATA_PATHS, TRAINING_CONFIG, MODEL_CONFIG
from data_utils import (load_labels, get_all_patch_files, group_patches_by_slice,
                        build_slice_to_class_map, split_by_case_stratified, build_case_dict, report_no_leak)
from models import create_model
from dataset import StainBagCaseDataset, case_collate_fn, create_transforms
from trainer import MILTrainer
from utils import (set_seed, get_device, save_data_splits, load_data_splits)

#define a hyperparameter tuner class that uses Optuna for Bayesian optimization of hyperparameters for the MIL model
class HyperparameterTuner:
    """
    Bayesian hyperparameter tuning using Optuna
    """
    
    def __init__(self, data_splits_path=None, n_trials=50, seed=42, study_name=None):
        """
        Args:
            data_splits_path: Path to existing data splits
            n_trials: Number of optimization trials
            seed: Random seed
            study_name: Name for the Optuna study
        """
        self.data_splits_path = data_splits_path
        self.n_trials = n_trials
        self.seed = seed
        self.study_name = study_name or f"mil_hyperparam_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = get_device()
        
        #create directory to save study results
        self.study_dir = os.path.join("runs", self.study_name)
        os.makedirs(self.study_dir, exist_ok=True)
        
        #object for storing training histories across trials
        self.trial_histories = {}
        
        #load and prepare data (same data used in every trial)
        self.train_data, self.val_data, self.test_data = self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for trials"""
        """Returns train, val, test data in the form of (case_dict, label_map) tuples"""
        print("=" * 60)
        print("PREPARING DATA FOR HYPERPARAMETER TUNING")
        print("=" * 60)
        
        #load labels
        labels = load_labels(DATA_PATHS['labels_csv'])
        print(f"Loaded {len(labels)} labels")
        
        #get patch files
        all_files = get_all_patch_files(DATA_PATHS['patches_dir'])
        print(f"Found {len(all_files)} patch files")
        
        #group patches by slice
        patches = group_patches_by_slice(all_files, DATA_PATHS['patches_dir'])
        print(f"Grouped into {len(patches)} slices")
        
        #build slice to class mapping
        slice_to_class = build_slice_to_class_map(patches, labels)
        print(f"Mapped {len(slice_to_class)} slices to classes")
        
        #group slices by class for stratified splitting
        slices_by_class = defaultdict(list)
        for key, label in slice_to_class.items():
            slices_by_class[label].append(key)
        
        print(f"Class distribution: {dict((k, len(v)) for k, v in slices_by_class.items())}")
        
        #load or create data splits
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
            #create new splits
            train_slices, val_slices, test_slices = split_by_case_stratified(
                slices_by_class, random_state=self.seed
            )
            
            #save splits for reproducibility
            train_cases = list(set([case_id for case_id, _ in train_slices]))
            val_cases = list(set([case_id for case_id, _ in val_slices]))
            test_cases = list(set([case_id for case_id, _ in test_slices]))
            save_data_splits(train_cases, val_cases, test_cases, self.study_dir)
        
        print(f"Split sizes - Train: {len(train_slices)}, Val: {len(val_slices)}, Test: {len(test_slices)}")
        
        #build case dictionaries
        train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)
        val_case_dict, val_label_map = build_case_dict(val_slices, patches, slice_to_class)
        test_case_dict, test_label_map = build_case_dict(test_slices, patches, slice_to_class)
        
        #verify no data leakage
        report_no_leak(train_case_dict, val_case_dict, test_case_dict)
        
        return (train_case_dict, train_label_map), (val_case_dict, val_label_map), (test_case_dict, test_label_map)
    
    def _create_data_loaders(self):
        """Create data loaders and datasets for tuning"""
        train_case_dict, train_label_map = self.train_data
        val_case_dict, val_label_map = self.val_data
        
        #create transforms
        train_transform = create_transforms(is_training=True)
        val_transform = create_transforms(is_training=False)
        
        #create datasets with trial parameters
        train_ds = StainBagCaseDataset(
            train_case_dict, train_label_map,
            transform=train_transform,
            per_slice_cap=MODEL_CONFIG['per_slice_cap'],
            max_slices_per_stain=MODEL_CONFIG['max_slices_per_stain'],
            shuffle_patches=True,
        )
        
        val_ds = StainBagCaseDataset(
            val_case_dict, val_label_map,
            transform=val_transform,
            per_slice_cap=MODEL_CONFIG['per_slice_cap'],
            max_slices_per_stain=MODEL_CONFIG['max_slices_per_stain'],
            shuffle_patches=True,
        )
        
        #create data loaders
        train_loader = DataLoader(
            train_ds, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True,
            num_workers=TRAINING_CONFIG['num_workers'], pin_memory=True, 
            collate_fn=case_collate_fn, persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_ds, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False,
            num_workers=TRAINING_CONFIG['num_workers'], pin_memory=True, 
            collate_fn=case_collate_fn, persistent_workers=True
        )
        
        return train_loader, val_loader
    
    def objective(self, trial):
        """
        Optuna objective function - defines the hyperparameter search space
        and returns the metric to optimize
        """
        #define hyperparameters to tune and their search space
        trial_params = {
            #model architecture
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            
            #training
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            
            #class weights for handling imbalance
            'class_weight_benign': trial.suggest_float('class_weight_benign', 1.0, 5.0),
            'class_weight_high_grade': 1.0,  #fixed as reference point
            
            #fixed parameters
            'epochs': 3,  #make high, will likely stop early
        }
        
        print(f"\n{'=' * 60}")
        print(f"Trial {trial.number}: {trial.params}")
        print(f"{'=' * 60}\n")
        
        #set seed for reproducibility
        set_seed(self.seed + trial.number)
        
        #create data loaders
        train_loader, val_loader = self._create_data_loaders()
        
        #create model with trial parameters
        model = create_model(
            num_classes=MODEL_CONFIG['num_classes'],
            embed_dim=MODEL_CONFIG['embed_dim'],
            dropout=trial_params['dropout']
        )
        
        #create checkpoint directory for this trial
        trial_checkpoint_dir = os.path.join(self.study_dir, f"trial_{trial.number}", "checkpoints")
        os.makedirs(trial_checkpoint_dir, exist_ok=True)
        
        #create trainer
        trainer = MILTrainer(model, self.device, checkpoint_dir=trial_checkpoint_dir)
        
        #update trainer parameters
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = trial_params['learning_rate']
            param_group['weight_decay'] = trial_params['weight_decay']
        
        #update class weights in loss function
        class_weights = torch.tensor([
            trial_params['class_weight_benign'],
            trial_params['class_weight_high_grade']
        ]).to(self.device)
        trainer.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        #train with pruning callback
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        #track training history for each trial
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'params': trial.params,
        }
        
        for epoch in range(trial_params['epochs']):
            #train epoch
            train_loss = trainer.train_epoch(train_loader)
            
            #validate
            val_loss, val_acc = trainer.validate(val_loader)

            #record epoch metrics
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['val_acc'].append(float(val_acc))
            
            #update scheduler
            if trainer.scheduler:
                if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    trainer.scheduler.step(val_loss)
                else:
                    trainer.scheduler.step()
            
            #report intermediate value for pruning
            trial.report(val_loss, epoch)
            
            #handle pruning
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}")
                history['pruned'] = True
                history['pruned_at_epoch'] = epoch
                self.trial_histories[trial.number] = history
                raise optuna.TrialPruned()
            
            #track best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            #early stopping
            if TRAINING_CONFIG['early_stopping'] and epochs_without_improvement >= TRAINING_CONFIG['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break
            
            print(f"Epoch {epoch + 1}/{trial_params['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        history['pruned'] = False
        history['best_val_loss'] = float(best_val_loss)
        self.trial_histories[trial.number] = history
        
        #clean up to free memory
        del model, trainer, train_loader, val_loader
        torch.cuda.empty_cache()
        
        #return best validation loss (this is what we are aiming to minimize)
        return best_val_loss
    
    def optimize(self):
        """Run the Bayesian optimization"""
        print(f"\n{'=' * 80}")
        print(f"STARTING BAYESIAN HYPERPARAMETER OPTIMIZATION")
        print(f"Study name: {self.study_name}")
        print(f"Number of trials: {self.n_trials}")
        print(f"{'=' * 80}\n")
        
        #create optuna study
        study = optuna.create_study(
            study_name=self.study_name,
            direction='minimize',  #minimize validation loss
            sampler=TPESampler(seed=self.seed, gamma=0.25),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        
        #run optimization
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        #print results
        print(f"\n{'=' * 80}")
        print("OPTIMIZATION COMPLETED")
        print(f"{'=' * 80}\n")
        
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation loss: {study.best_value:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        #save results
        self._save_results(study)
        
        return study
    
    def _save_results(self, study):
        """Save optimization results"""
        #save best parameters as JSON
        best_params_path = os.path.join(self.study_dir, "best_hyperparameters.json")
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        print(f"\nBest hyperparameters saved to: {best_params_path}")
        
        #save all trials as CSV
        trials_df = study.trials_dataframe()
        trials_csv_path = os.path.join(self.study_dir, "all_trials.csv")
        trials_df.to_csv(trials_csv_path, index=False)
        print(f"All trials saved to: {trials_csv_path}")

        #save all training histories as JSON
        histories_path = os.path.join(self.study_dir, "trial_histories.json")
        with open(histories_path, 'w') as f:
            #convert int keys to strings for valid JSON
            json.dump({str(k): v for k, v in self.trial_histories.items()}, f, indent=2)
        print(f"Training histories saved to: {histories_path}")
        
        #save study summary
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
        
        #create visualizations
        try:

            #feature importances 
            fig = vis.plot_param_importances(study)
            fig.write_image(os.path.join(self.study_dir, "param_importances.png"))

            #trial convergence: val_loss curves for every trial (completed only)
            completed_trials = {
                k: v for k, v in self.trial_histories.items()
                if not v.get('pruned', False) and v.get('val_loss')
            }

            best_trial_num = study.best_trial.number

            fig, ax = plt.subplots(figsize=(10, 6))

            colormap = cm.viridis(np.linspace(0, 1, max(len(completed_trials), 1)))

            for idx, (trial_num, hist) in enumerate(sorted(completed_trials.items())):
                epochs = list(range(1, len(hist['val_loss']) + 1))
                is_best = (trial_num == best_trial_num)
                ax.plot(
                    epochs, hist['val_loss'],
                    color='#FF5050' if is_best else colormap[idx],
                    linewidth=2.5 if is_best else 1.2,
                    alpha=1.0 if is_best else 0.7,
                    label=f"Trial {trial_num} ★ best" if is_best else f"Trial {trial_num}",
                    zorder=3 if is_best else 2,
                )

            ax.set_title("Validation Loss Convergence — Completed Trials", fontsize=13)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Loss")
            ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.7)
            ax.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(self.study_dir, "trial_convergence.png"), dpi=150)
            plt.close(fig)

            #train vs val loss curves per trial
            all_finished = {k: v for k, v in self.trial_histories.items() if v.get('val_loss')}
            n_finished = len(all_finished)

            if n_finished > 0:
                n_cols = min(4, n_finished)
                n_rows = math.ceil(n_finished / n_cols)

                fig, axes = plt.subplots(
                    n_rows, n_cols,
                    figsize=(4.5 * n_cols, 3 * n_rows),
                    squeeze=False
                )

                for idx, (trial_num, hist) in enumerate(sorted(all_finished.items())):
                    row, col = idx // n_cols, idx % n_cols
                    ax = axes[row][col]
                    epochs = list(range(1, len(hist['val_loss']) + 1))

                    ax.plot(epochs, hist['train_loss'], color='steelblue', linewidth=1.5, label='train')
                    ax.plot(epochs, hist['val_loss'],   color='tomato',    linewidth=1.5, label='val')

                    title = f"Trial {trial_num}" + (" ★" if trial_num == best_trial_num else "")
                    ax.set_title(title, fontsize=9)
                    ax.set_xlabel("Epoch", fontsize=7)
                    ax.set_ylabel("Loss", fontsize=7)
                    ax.tick_params(labelsize=6)
                    ax.grid(True, linestyle='--', alpha=0.4)
                    if idx == 0:
                        ax.legend(fontsize=7)

                #hide unused subplots
                for idx in range(n_finished, n_rows * n_cols):
                    axes[idx // n_cols][idx % n_cols].set_visible(False)

                fig.suptitle("Train vs Validation Loss — Per Trial", fontsize=13, y=1.01)
                plt.tight_layout()
                plt.savefig(os.path.join(self.study_dir, "per_trial_loss_curves.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)

            #parameter vs performance scatter plots
            completed_optuna = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_optuna:
                param_names = list(completed_optuna[0].params.keys())
                trial_values = [t.value for t in completed_optuna]

                n_params = len(param_names)
                n_cols_p = min(3, n_params)
                n_rows_p = math.ceil(n_params / n_cols_p)

                fig, axes = plt.subplots(
                    n_rows_p, n_cols_p,
                    figsize=(5 * n_cols_p, 4 * n_rows_p),
                    squeeze=False
                )

                norm = plt.Normalize(vmin=min(trial_values), vmax=max(trial_values))
                cmap = plt.cm.RdYlGn_r

                for idx, param in enumerate(param_names):
                    row, col = idx // n_cols_p, idx % n_cols_p
                    ax = axes[row][col]

                    x_vals, y_vals = [], []
                    for t in completed_optuna:
                        if param in t.params:
                            x_vals.append(float(t.params[param]))
                            y_vals.append(t.value)

                    sc = ax.scatter(
                        x_vals, y_vals,
                        c=y_vals, cmap=cmap, norm=norm,
                        s=60, edgecolors='grey', linewidths=0.4, zorder=3
                    )
                    if idx == 0:
                        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label("Val Loss", fontsize=7)

                    ax.set_title(param, fontsize=9)
                    ax.set_xlabel(param, fontsize=8)
                    ax.set_ylabel("Val Loss", fontsize=8)
                    ax.tick_params(labelsize=6)
                    ax.grid(True, linestyle='--', alpha=0.4)

                # hide unused subplots
                for idx in range(n_params, n_rows_p * n_cols_p):
                    axes[idx // n_cols_p][idx % n_cols_p].set_visible(False)

                fig.suptitle("Hyperparameter Values vs Validation Loss", fontsize=13, y=1.01)
                plt.tight_layout()
                plt.savefig(os.path.join(self.study_dir, "param_vs_performance.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)

            print(f"Visualizations saved to: {self.study_dir}")

        except ImportError:
            print("Install plotly for visualizations: pip install matplotlib")


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
    
    #create tuner
    tuner = HyperparameterTuner(
        data_splits_path=args.data_splits,
        n_trials=args.n_trials,
        seed=args.seed,
        study_name=args.study_name
    )
    
    #eun optimization
    study = tuner.optimize()
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()