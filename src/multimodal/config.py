"""Configuration for multimodal fusion model."""

import pathlib

# Paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Random seeds for repeated random subsampling
RANDOM_SEEDS = list(range(10))

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Feature dimensions
FEATURE_DIMS = {
    "rdkit": 134,        # RDKit descriptors (after preprocessing)
    "chemberta": 768,    # ChemBERTa [CLS] embedding dimension
    "graph": 256,        # Graph feature dimension after GCN pooling
}

# ChemBERTa model configuration
CHEMBERTA_CONFIG = {
    "model_name": "seyonec/PubChem10M_SMILES_BPE_450k",
    "max_length": 128,
    "freeze": True,  # Freeze ChemBERTa weights during training
}

# Model architecture configuration
MODEL_CONFIG = {
    "rdkit_hidden": None,  # None = direct pass-through
    "chemberta_projection": 384,  # Project 768 -> 384
    "graph_hidden_dims": [128, 256],  # GCN layer dimensions
    "fusion_dims": [512, 256],  # MLP after concatenation
    "dropout": 0.2,
}

# Training parameters
TRAINING_PARAMS = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "nb_epoch": 100,
    "early_stopping_patience": 15,
    "early_stopping_metric": "roc_auc",
    "weight_decay": 1e-4,
}

# Hyperparameter tuning
OPTUNA_N_TRIALS = 50
BEST_PARAMS_FILENAME = "best_params.json"

# Optuna search space for multimodal fusion model
MULTIMODAL_SEARCH_SPACE = {
    "learning_rate": ("float_log", 0.0001, 0.01),
    "batch_size": ("categorical", [32, 64, 128]),
    "dropout": ("float", 0.0, 0.5),
    "chemberta_projection": ("int", 256, 512),
    "graph_hidden_dims": ("categorical", [
        [64, 128],
        [128, 256],
        [256, 512],
    ]),
    "fusion_dims": ("categorical", [
        [256, 128],
        [512, 256],
        [768, 384],
    ]),
    "weight_decay": ("float_log", 1e-5, 1e-3),
}

# Default configuration
DEFAULT_CONFIG_KEY = "Multimodal_Fusion"
