"""
Configuration file for MIL training
"""
import os
from typing import Tuple

# Data paths (adjust these for your HPC environment)
DATA_PATHS = {
    'labels_csv': '/home/sml7045/presentation3/extra_benign_case_grade_match.csv',
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



#Updating with the most recent Hyperparams from OPTUNA Study
# Training configuration
# Training configuration
TRAINING_CONFIG = {
    'epochs': 30,  # Increased since we have early stopping
    'batch_size': 1,  # MIL typically uses batch_size=1
    'learning_rate': 0.00024106495902171608,  # Updated from optuna_tuning_feb26
    'weight_decay': 7.652872182750095e-05,  # Updated from optuna_tuning_feb26
    'num_workers': 2,
    'pin_memory': True,
    'random_state': 42,
    'class_weights': [3.3789978831283785, 1.0],  # Updated benign weight from optuna_tuning_feb26
    'patch_proj_dropout': 0.49087538832936756,   # Updated from optuna_tuning_feb26
    'classifier_dropout': 0.43253984700833437,  # Updated from optuna_tuning_feb26
    'entropy_lambda': 0.006161049539380964, # Updated from optuna_tuning_feb26
    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_type': 'reduce_on_plateau',  # 'reduce_on_plateau' or 'cosine'
    'scheduler_patience': 4,  # For ReduceLROnPlateau
    'scheduler_factor': 0.2139,  # Reduce LR by half
    'scheduler_min_lr': 1e-6,
    # Early stopping
    'early_stopping': False, #get thirty epochs on best model
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
