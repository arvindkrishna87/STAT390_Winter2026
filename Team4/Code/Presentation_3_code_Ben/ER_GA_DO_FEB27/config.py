"""
Configuration file for MIL training
"""
import os
from typing import Tuple

# Data paths (adjust these for your HPC environment)
DATA_PATHS = {
    'labels_csv': '/projects/e32998/STAT390_Krish/Code/Code4_reduce_runtime/extra_benign_case_grade_match.csv',
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

#Best hyperaparemters from the most recent OPTUNA study
OPTUNA_BEST_PARAMS = {
    "learning_rate": 7.309539835912905e-05,
    "weight_decay": 7.476312062252303e-05,
    "dropout": 0.34474115788895177,
    "class_weight_benign": 1.8487346516301046,
    "scheduler_min_lr": 1.2562773503807019e-06, 
    "entropy_lambda": 0.00014742753159914678
}

#Best hyperparameters from the most recent OPTUNA study2
OPTUNA_BEST_PARAMS_2 = {
    "learning_rate": 6.798962421591133e-05,
    "weight_decay": 2.6587543983272695e-05,
    "dropout": 0.28636499344142013,
    "class_weight_benign": 1.6283831568974036, 
    "scheduler_min_lr": 7.790143126276235e-07,
    "entropy_lambda": 4.059611610484306e-05
}

#Updating with the most recent Hyperparams from OPTUNA Study
# Training configuration
# Training configuration
TRAINING_CONFIG = {
    'epochs': 30,  # Increased since we have early stopping
    'batch_size': 1,  # MIL typically uses batch_size=1
    'learning_rate': OPTUNA_BEST_PARAMS_2['learning_rate'],
    'weight_decay': OPTUNA_BEST_PARAMS_2['weight_decay'],
    'num_workers': 2,
    'pin_memory': True,
    'random_state': 42,
    'class_weights': [OPTUNA_BEST_PARAMS_2['class_weight_benign'], 1.0],  # Updated from OPTUNA study
    'dropout': OPTUNA_BEST_PARAMS_2['dropout'],  # Add dropout for regularization
    'entropy_lambda': OPTUNA_BEST_PARAMS_2['entropy_lambda'],  # Add entropy regularization
    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_type': 'reduce_on_plateau',  # 'reduce_on_plateau' or 'cosine'
    'scheduler_patience': 6,  # For ReduceLROnPlateau
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
