# Best hyperparameters found by Optuna
# Copy these values to your config.py

TRAINING_CONFIG = {
    'learning_rate': 1.3066739238053272e-05,
    'weight_decay': 0.0003967605077052988,
    'coeff_dropout': 0.3005575058716044,        # patch projector dropout
    'dropout': 0.35403628889802274,          # classifier head dropout
    'class_weights': [1.0823379771832098, 1.0],
    'entropy_coeff': 0.48495492608099716,
}
