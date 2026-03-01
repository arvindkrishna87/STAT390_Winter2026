"""
Configuration file for MIL training
"""
import os
from typing import Tuple

# Data paths (adjust these for your HPC environment)
DATA_PATHS = {
    'labels_csv': '/home/bdp1083/Presentation_3_code/Updated_Model_code/extra_benign_case_grade_match.csv',
    'patches_dir': '/projects/e32998/patches_benign_split',
    'runs_dir': '/projects/e32998/MIL_training/final_runs'  # Base directory for training runs
}

# Model configuration
MODEL_CONFIG = {
    'num_classes': 2,
    'embed_dim': 512,
    'attention_hidden_dim': 128,
    'per_slice_cap': 800,
    'max_slices_per_stain': None,
    'stains': ('h&e', 'melan', 'sox10')
}


#Hyperparameters from the most recent OPTUNA study 1
OPTUNA_BEST_PARAMS_1 = {
    "learning_rate": 1.0994335574766187e-05,
    "weight_decay": 0.008123245085588688,
    "dropout": 0.4329770563201687,
    "class_weight_benign": 2.0308477766956905,
    "scheduler_min_lr": 3.511356313970404e-07}

OPTUNA_BEST_PARAMS_2 = {
    "learning_rate": 5.2426938625973105e-06,
    "weight_decay": 0.018890282808631323,
    "dropout": 0.21648852816008435,
    "class_weight_benign": 2.012339110678276, 
    "scheduler_min_lr": 1.5199348301309797e-07
}

#Updating with the most recent Hyperparams from OPTUNA Study
# Training configuration
# Training configuration
TRAINING_CONFIG = {
    'epochs': 30,  # Increased since we have early stopping
    'batch_size': 1,  # MIL typically uses batch_size=1
    'learning_rate': OPTUNA_BEST_PARAMS_2['learning_rate'],  # Updated from OPTUNA study
    'weight_decay': OPTUNA_BEST_PARAMS_2['weight_decay'],  # Updated from OPTUNA study
    'num_workers': 2,
    'pin_memory': True,
    'random_state': 42,
    'class_weights': [OPTUNA_BEST_PARAMS_2['class_weight_benign'], 1.0],  # Increased benign weight from 2.0 to 2.5285 (from OPTUNA study)
    'dropout': OPTUNA_BEST_PARAMS_2['dropout'],  # Add dropout for regularization
    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_type': 'reduce_on_plateau',  # 'reduce_on_plateau' or 'cosine'
    'scheduler_patience': 4,  # For ReduceLROnPlateau
    'scheduler_factor': 0.2139,  # Reduce LR by half
    'scheduler_min_lr': OPTUNA_BEST_PARAMS_2['scheduler_min_lr'],  # Updated from OPTUNA study
    # Early stopping
    'early_stopping': True,
    'early_stopping_patience': 8,  # Stop if no improvement for 10 epochs
    'early_stopping_min_delta': 0.001,  # Minimum change to qualify as improvement
    'early_stopping_min_epochs': 10  # Minimum epochs before early stopping can trigger
}


# Data split configuration
SPLIT_CONFIG = {
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'stratify': True
}

# Image preprocessing
IMAGE_CONFIG = {
    'image_size': (224, 224),
    'normalize_mean': [0.485, 0.456, 0.406],  # DenseNet mean
    'normalize_std': [0.229, 0.224, 0.225]   # DenseNet std
}

# Valid classes for filtering
VALID_CLASSES = [1.0, 3.0, 4.0]

# Device configuration
DEVICE = 'cuda' if os.environ.get('CUDA_AVAILABLE', 'true').lower() == 'true' else 'cpu'
