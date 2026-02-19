#!/usr/bin/env python3
"""
Script for Optuna hyperparameter optimization, edited from main.py
"""
import os
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
import optuna

# Import our modules
from config import DATA_PATHS, TRAINING_CONFIG, MODEL_CONFIG
from data_utils import (
   load_labels, get_all_patch_files, group_patches_by_slice,
   build_slice_to_class_map, split_by_case_stratified, build_case_dict,
   report_no_leak, summarize_case_dict
)
#from models import create_model
from opt_models import create_model2
from dataset import StainBagCaseDataset, case_collate_fn, create_transforms
from trainer import MILTrainer, count_patches_by_class
from opt_trainer import MILTrainer2
from utils import (
   set_seed, get_device, print_data_summary, create_run_directory,
   save_data_splits, load_data_splits, print_model_summary, check_data_integrity
)
from attention_analysis import analyze_attention_weights


def parse_args():
   """Parse command line arguments"""
   parser = argparse.ArgumentParser(description='Train Hierarchical Attention MIL model')
  
   # Data arguments
   parser.add_argument('--labels_csv', type=str, default=DATA_PATHS['labels_csv'],
                      help='Path to labels CSV file')
   parser.add_argument('--patches_dir', type=str, default=DATA_PATHS['patches_dir'],
                      help='Path to patches directory')
   # checkpoint_dir is now automatically set to {run_dir}/checkpoints
  
   # Training arguments
   parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'],
                      help='Number of training epochs')
   parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'],
                      help='Learning rate')
   parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'],
                      help='Batch size (typically 1 for MIL)')
   parser.add_argument('--num_workers', type=int, default=TRAINING_CONFIG['num_workers'],
                      help='Number of data loader workers')
  
   # Model arguments
   parser.add_argument('--embed_dim', type=int, default=MODEL_CONFIG['embed_dim'],
                      help='Embedding dimension')
   parser.add_argument('--per_slice_cap', type=int, default=MODEL_CONFIG['per_slice_cap'],
                      help='Maximum patches per slice')
   parser.add_argument('--max_slices_per_stain', type=int, default=MODEL_CONFIG['max_slices_per_stain'],
                      help='Maximum slices per stain (None for unlimited)')
  
   # Other arguments
   parser.add_argument('--seed', type=int, default=TRAINING_CONFIG['random_state'],
                      help='Random seed')
   parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
   parser.add_argument('--eval_only', action='store_true',
                      help='Only evaluate, do not train')
   parser.add_argument('--analyze_attention', action='store_true',
                      help='Perform attention analysis and visualization')
   parser.add_argument('--attention_top_n', type=int, default=5,
                      help='Number of top/bottom patches to visualize')
   parser.add_argument('--load_splits', type=str, default=None,
                      help='Path to data_splits.npz file to load existing splits')

   return parser.parse_args()

class ModelHyperparameters:
   def __init__(self, embed_dim, attention_hidden_dim, epochs, lr, batch_size,
            dropout=0.2, weight_decay=0.0, class_weights=[1,1,1]):
      # Architecture
      self.embed_dim = embed_dim
      self.attention_hidden_dim = attention_hidden_dim
      self.per_slice_cap = 100 
      self.max_slices_per_stain = 10
      self.dropout = dropout
      # Training
      self.learning_rate = lr
      self.weight_decay = weight_decay
      self.epochs = epochs
      self.batch_size = batch_size
      self.class_weights = class_weights

def choose_parameters(trial):
  # Architecture
  embed_dim = trial.suggest_categorical('embed_dim', [128, 256, 512]) #currently set at 512
  attention_hidden_dim = trial.suggest_categorical('attention_hidden_dim', [64, 128, 256]) #currently set at 128
  dropout = trial.suggest_float('dropout', 0.1, 0.5) #currently set at 0.3 at three places
  # Training 
  learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True) #currently set at 3e-4
  weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True) #currently set at 2e-4
  #per_slice_cap = 100 
  #max_slices_per_stain = 10 
  #epochs = trial.suggest_int('epochs', 5, 20)
  #batch_size = trial.suggest_categorical('batch_size', [1, 2, 4])
  #class_weights = [
      #trial.suggest_float('w1', 0.5, 2.0),
      #trial.suggest_float('w2', 0.5, 2.0),
      #trial.suggest_float('w3', 0.5, 2.0)
  #]
  return ModelHyperparameters(
      embed_dim=embed_dim,
      attention_hidden_dim=attention_hidden_dim,
      #epochs=epochs,
      epochs=TRAINING_CONFIG['epochs'],
      lr=learning_rate,
      batch_size=1,
      dropout=dropout,
      weight_decay=weight_decay,
      class_weights=TRAINING_CONFIG['class_weights']
      )

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
       test_cases_set = set(splits_data['test_cases'])
      
       # Map loaded case IDs back to slices
       train_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in train_cases_set]
       val_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in val_cases_set]
       test_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in test_cases_set]
      
       print(f"Loaded splits - Train: {len(train_slices)}, Val: {len(val_slices)}, Test: {len(test_slices)}")
   else:
       # Split data by case (stratified)
       train_slices, val_slices, test_slices = split_by_case_stratified(
           slices_by_class, random_state=args.seed
       )
      
       print(f"Split sizes - Train: {len(train_slices)}, Val: {len(val_slices)}, Test: {len(test_slices)}")
  
   # Build case dictionaries
   train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)
   val_case_dict, val_label_map = build_case_dict(val_slices, patches, slice_to_class)
   test_case_dict, test_label_map = build_case_dict(test_slices, patches, slice_to_class)
  
   # Check for data leakage
   report_no_leak(train_case_dict, val_case_dict, test_case_dict)
  
   # Create summary DataFrames
   train_df = summarize_case_dict(train_case_dict, train_label_map, "train")
   val_df = summarize_case_dict(val_case_dict, val_label_map, "val")
   test_df = summarize_case_dict(test_case_dict, test_label_map, "test")
  
   # Print data summary
   print_data_summary(train_df, val_df, test_df)
  
   # Count patches by class
   count_patches_by_class(train_case_dict, train_label_map, "Train")
   count_patches_by_class(val_case_dict, val_label_map, "Validation")
   count_patches_by_class(test_case_dict, test_label_map, "Test")
  
   # Check data integrity
   check_data_integrity(train_case_dict, train_label_map, "Train")
   check_data_integrity(val_case_dict, val_label_map, "Validation")
   check_data_integrity(test_case_dict, test_label_map, "Test")
  
   return (train_case_dict, train_label_map), (val_case_dict, val_label_map), (test_case_dict, test_label_map)



def create_data_loaders(train_data, val_data, test_data, args):
   """Create data loaders"""
   print("\n" + "=" * 60)
   print("CREATING DATA LOADERS")
   print("=" * 60)
  
   train_case_dict, train_label_map = train_data
   val_case_dict, val_label_map = val_data
   test_case_dict, test_label_map = test_data
  
   # Create transforms
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
       shuffle_patches=True,  # Enable shuffling for better sampling
   )
  
   test_ds = StainBagCaseDataset(
       test_case_dict, test_label_map,
       transform=val_transform,
       per_slice_cap=args.per_slice_cap,
       max_slices_per_stain=args.max_slices_per_stain,
       shuffle_patches=True,  # Enable shuffling for better sampling
   )
  
   # Create data loaders
   train_loader = DataLoader(
       train_ds, batch_size=args.batch_size, shuffle=True,
       num_workers=args.num_workers, pin_memory=True, collate_fn=case_collate_fn,
       persistent_workers=True
   )
  
   val_loader = DataLoader(
       val_ds, batch_size=args.batch_size, shuffle=False,
       num_workers=args.num_workers, pin_memory=True, collate_fn=case_collate_fn,
       persistent_workers=True
   )
  
   test_loader = DataLoader(
       test_ds, batch_size=args.batch_size, shuffle=False,
       num_workers=args.num_workers, pin_memory=True, collate_fn=case_collate_fn,
       persistent_workers=True
   )
  
   print(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
  
   return train_loader, val_loader, test_loader

#just tuning embed_dim, attention_hidden_dim, dropout learning_rate, weight_decay

def objective(trial, train_loader, val_loader, device):
   params = choose_parameters(trial)
   model = create_model2(
       num_classes=MODEL_CONFIG['num_classes'],
       embed_dim=params.embed_dim,
       attention_hidden_dim=params.attention_hidden_dim,
       dropout=params.dropout
   )
   trainer = MILTrainer2(
       model=model,
       device=device,
       learning_rate=params.learning_rate,
       weight_decay=params.weight_decay,
       checkpoint_dir=None 
   )
   epochs = 3  #small for first run
   best_val_loss = float("inf")

   for epoch in range(epochs):
       trainer.train_epoch(train_loader)
       val_loss, _ = trainer.validate(val_loader)

       trial.report(val_loss, epoch)

       if trial.should_prune():
           raise optuna.TrialPruned()
       if val_loss < best_val_loss:
           best_val_loss = val_loss

   return best_val_loss


def main():
   args = parse_args()

   set_seed(args.seed)
   device = get_device()

   train_data, val_data, test_data = prepare_data(args)

   train_loader, val_loader, _ = create_data_loaders(
       train_data, val_data, test_data, args
   )
   study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2))  # waiting for 2 trials to complete

   study.optimize(
       lambda trial: objective(trial, train_loader, val_loader, device),
       n_trials=7 
   )

   print("\n" + "=" * 60)
   print("OPTUNA FINISHED")
   print("=" * 60)

   print(f"Best Validation Loss: {study.best_value}")
   print("Best Hyperparameters:")
   for key, value in study.best_params.items():
       print(f"  {key}: {value}")


if __name__ == "__main__":
   main()