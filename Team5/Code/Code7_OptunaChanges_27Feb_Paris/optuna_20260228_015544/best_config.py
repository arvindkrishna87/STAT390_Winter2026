# Best hyperparameters found by Optuna
# Copy these values to your config.py

TRAINING_CONFIG = {
    'learning_rate': 2.7528862695812924e-05,
    'weight_decay': 4.044574059404959e-05,
    'coeff_dropout': 0.493388667319079,        # patch projector dropout
    'dropout': 0.3930373854314628,          # classifier head dropout
    'class_weights': [4.160322358622171, 1.0],
    'entropy_coeff': 0.4990497566649651,
}
