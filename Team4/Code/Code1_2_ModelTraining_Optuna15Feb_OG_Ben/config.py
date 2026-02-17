"""
Configuration file for MIL training
"""
import os
from typing import Tuple

# Data paths (adjust these for your HPC environment)
DATA_PATHS = {
    'labels_csv': '/projects/e32998/Fall2025_arXiv/MIL_training/case_grade_match.csv', #updating path to case_grade_match.csv
    'patches_dir': '/projects/e32998/patches',
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

BEST_PARAMS = {
        "learning_rate": 0.00026176655097040057,
        "weight_decay": 0.0004176805377655882,
        "dropout": 0.22654775061557586,
        "class_weight_benign": 1.8919657248382904,
        "scheduler_patience": 2,
        "scheduler_factor": 0.2975990992289793
    }


# Training configuration
# Training configuration
TRAINING_CONFIG = {
    'epochs': 30,  # Increased since we have early stopping
    'batch_size': 1,  # MIL typically uses batch_size=1
    'learning_rate': BEST_PARAMS["learning_rate"],  # Use best params for learning rate
    'weight_decay': BEST_PARAMS["weight_decay"],  # Use best params for weight decay
    'num_workers': 2,
    'pin_memory': True,
    'random_state': 42,
    'class_weights': [BEST_PARAMS["class_weight_benign"], 1.0],  # Use best params for class weights
    'dropout': BEST_PARAMS["dropout"],  # Use best params for dropout
    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_type': 'reduce_on_plateau',  # 'reduce_on_plateau' or 'cosine'
    'scheduler_patience': BEST_PARAMS["scheduler_patience"],  # Use best params for scheduler patience
    'scheduler_factor': BEST_PARAMS["scheduler_factor"],  # Use best params for scheduler factor
    'scheduler_min_lr': 1e-6,
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
