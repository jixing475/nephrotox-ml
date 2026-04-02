"""Configuration for DeepChem GCN model."""

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Random seeds for repeated random subsampling (supports up to 50 iterations)
RANDOM_SEEDS = list(range(50))

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Backend/工具定义
BACKENDS = {
    "DeepChem": "deepchem",
    "DGLlife": "dgllife",
    "Chemprop": "chemprop",
}

# 特征集定义
FEATURE_SETS = {
    "Graph": {"type": "graph", "extra_features": None},
    "ChemoPy2d": {"type": "descriptor", "file_suffix": "chemopy2d"},
    "RDKit": {"type": "descriptor", "file_suffix": "rdkit_desc"},
    "Graph+ChemoPy2d": {"type": "graph+descriptor", "extra_features": "chemopy2d"},
}

# 模型配置 - 统一命名规范: {Algorithm}_{Tool}_{Feature}
MODEL_CONFIGS = {
    # DeepChem 后端
    "GCN_DeepChem_Graph": {
        "algorithm": "GCN",
        "backend": "deepchem",
        "model_class": "GCNModel",
        "feature_type": "Graph",
        "train_file": INPUT_DIR / "cleaned_data_train_rdkit_desc.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_rdkit_desc.csv",
    },
    
    # DGLlife 后端
    "GCN_DGLlife_Graph": {
        "algorithm": "GCN",
        "backend": "dgllife",
        "model_class": "DGLlifeGCNClassifier",
        "feature_type": "Graph",
        "train_file": INPUT_DIR / "cleaned_data_train_rdkit_desc.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_rdkit_desc.csv",
    },
    "GAT_DGLlife_Graph": {
        "algorithm": "GAT",
        "backend": "dgllife",
        "model_class": "GATPredictor",
        "feature_type": "Graph",
        "train_file": INPUT_DIR / "cleaned_data_train_rdkit_desc.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_rdkit_desc.csv",
    },
    "Weave_DGLlife_Graph": {
        "algorithm": "Weave",
        "backend": "dgllife",
        "model_class": "WeavePredictor",
        "feature_type": "Graph",
        "train_file": INPUT_DIR / "cleaned_data_train_rdkit_desc.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_rdkit_desc.csv",
    },
    "AttentiveFP_DGLlife_Graph": {
        "algorithm": "AttentiveFP",
        "backend": "dgllife",
        "model_class": "AttentiveFPPredictor",
        "feature_type": "Graph",
        "train_file": INPUT_DIR / "cleaned_data_train_rdkit_desc.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_rdkit_desc.csv",
    },
    
    # Chemprop backend - D-MPNN
    "DMPNN_Chemprop_Graph": {
        "algorithm": "DMPNN",
        "backend": "chemprop",
        "model_class": "MPNN",
        "feature_type": "Graph",
        "train_file": INPUT_DIR / "cleaned_data_train_rdkit_desc.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_rdkit_desc.csv",
    },
    "DMPNN_Chemprop_Graph+RDKit": {
        "algorithm": "DMPNN",
        "backend": "chemprop",
        "model_class": "MPNN",
        "feature_type": "Graph+RDKit",
        "train_file": INPUT_DIR / "cleaned_data_train_rdkit_desc.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_rdkit_desc.csv",
    },
    "DMPNN_Chemprop_Graph+ChemoPy2d": {
        "algorithm": "DMPNN",
        "backend": "chemprop",
        "model_class": "MPNN",
        "feature_type": "Graph+ChemoPy2d",
        "train_file": INPUT_DIR / "cleaned_data_train_chemopy2d.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_chemopy2d.csv",
    },
}

DEFAULT_CONFIG_KEY = "GCN_DeepChem_Graph"

# Hyperparameter tuning defaults
# OPTIMIZED: Increased trials for more thorough search (Priority 7)
OPTUNA_N_TRIALS = 100  # Doubled from 50 for better hyperparameter exploration
TUNING_CV_FOLDS = 3    # Not used for DL (single train/val split), kept for consistency
BEST_PARAMS_FILENAME = "best_params.json"

# ============================================================================
# SEARCH SPACES - OPTIMIZED for broader search & enhanced regularization
# Key changes:
#   1. dropout ranges expanded to 0.0-0.5 (Priority 4: stronger regularization)
#   2. learning_rate ranges widened to cover paper params
#   3. weight_decay ranges expanded
#   4. More layer architecture options added
# ============================================================================

# Optuna search space for GCN model (DeepChem)
GCN_SEARCH_SPACE = {
    "learning_rate": ("float_log", 0.0005, 0.15),    # Wider range: covers paper's 0.09
    "batch_size": ("categorical", [32, 64, 128, 256]),  # Added 32
    "dropout": ("float", 0.0, 0.5),                  # ENHANCED: 0.3→0.5 for stronger regularization
    "graph_conv_layers": ("categorical", [
        [32, 32, 32],       # Smaller option
        [64, 64, 64],       # Paper's 3 layers
        [128, 128, 128],
        [256, 256, 256],
        [64, 64],           # 2 layers
        [128, 128],
        [64, 128, 64],      # Bottleneck architecture
    ]),
    "dense_layer_size": ("int", 32, 512),            # Lower bound reduced
    "weight_decay": ("float_log", 1e-5, 0.05),       # ENHANCED: wider range for stronger regularization
}

# DGLlife GCN search space
DGLLIFE_GCN_SEARCH_SPACE = {
    "learning_rate": ("float_log", 0.0005, 0.15),    # Wider range: covers paper's 0.09
    "batch_size": ("categorical", [32, 64, 128, 256]),  # Added 32
    "dropout": ("float", 0.0, 0.5),                  # ENHANCED: 0.3→0.5 for stronger regularization
    "hidden_feats": ("categorical", [
        [32, 32, 32],       # Smaller option
        [64, 64, 64],       # Paper's 3 layers
        [128, 128, 128],
        [256, 256, 256],
        [64, 64],           # 2 layers
        [128, 128],
        [64, 128, 64],      # Bottleneck architecture
    ]),
    "classifier_hidden_feats": ("int", 32, 512),     # Lower bound reduced
    "weight_decay": ("float_log", 1e-5, 0.05),       # ENHANCED: wider range for stronger regularization
}

# DGLlife GAT search space
DGLLIFE_GAT_SEARCH_SPACE = {
    "learning_rate": ("float_log", 0.0005, 0.05),    # Paper uses 0.001
    "batch_size": ("categorical", [32, 64, 128, 256]),
    "dropout": ("float", 0.0, 0.5),                  # ENHANCED: stronger regularization
    "hidden_feats": ("categorical", [
        [32, 32, 32],
        [64, 64, 64],       # Paper's 3 layers
        [128, 128, 128],
        [64, 64],
        [128, 128],
    ]),
    "num_heads": ("categorical", [
        [4, 4, 4],
        [6, 6, 6],          # Paper uses 6 heads
        [8, 8, 8],
        [4, 4],
        [6, 6],
        [8, 8],
    ]),
    "classifier_hidden_feats": ("int", 32, 512),
    "predictor_hidden_feats": ("int", 32, 512),
    "weight_decay": ("float_log", 1e-5, 0.05),       # ENHANCED
}

# DGLlife Weave search space
DGLLIFE_WEAVE_SEARCH_SPACE = {
    "learning_rate": ("float_log", 0.0005, 0.05),    # Paper uses 0.009
    "batch_size": ("categorical", [32, 64, 128, 256]),
    "num_gnn_layers": ("int", 2, 5),                 # Paper uses 4, expanded to 5
    "gnn_hidden_feats": ("int", 32, 150),            # Wider range
    "graph_feats": ("int", 64, 300),                 # Wider range
    "weight_decay": ("float_log", 1e-5, 0.01),       # Paper uses 0.0005
}

# DGLlife AttentiveFP search space
DGLLIFE_ATTENTIVEFP_SEARCH_SPACE = {
    "learning_rate": ("float_log", 0.0003, 0.02),    # Paper uses 0.0015
    "batch_size": ("categorical", [32, 64, 128, 256]),
    "dropout": ("float", 0.1, 0.5),                  # ENHANCED: Paper uses 0.32, force higher dropout
    "num_layers": ("int", 2, 5),                     # Paper uses 3
    "num_timesteps": ("categorical", [1, 2, 3, 4]),  # Paper uses 1
    "graph_feat_size": ("int", 100, 400),            # Wider range
    "weight_decay": ("float_log", 1e-5, 0.05),       # ENHANCED
}

# Chemprop D-MPNN search space
# OPTIMIZED: Expanded to match paper's hidden_size=2400
CHEMPROP_DMPNN_SEARCH_SPACE = {
    "learning_rate": ("float_log", 5e-5, 0.01),
    "batch_size": ("categorical", [32, 64, 128]),
    "depth": ("int", 2, 6),                          # Paper uses depth=6
    "hidden_dim": ("int", 300, 2500),                # EXPANDED: paper uses 2400
    "ffn_hidden_dim": ("int", 300, 2500),            # EXPANDED: paper uses 2400
    "ffn_num_layers": ("int", 1, 4),                 # Paper uses 3
    "dropout": ("float", 0.0, 0.5),                  # ENHANCED
    "weight_decay": ("float_log", 1e-6, 0.01),       # ENHANCED
}

# Fixed training parameters during tuning
# OPTIMIZED: Increased patience for better convergence (Priority 4)
TRAINING_PARAMS = {
    "nb_epoch": 300,                    # INCREASED: 200→300 for more training time
    "early_stopping_patience": 25,      # INCREASED: 10→25 to avoid premature stopping
    "early_stopping_metric": "roc_auc", # Monitoring metric: "roc_auc" or "loss"
    "eval_interval": 5,                 # Evaluation interval (every N epochs)
    "save_best_model": True,            # Whether to save best model
}

# Feature engineering parameters for fusion mode (Graph+Descriptors)
DESCRIPTOR_PREPROCESSING = {
    "variance_threshold": 0.0,          # Remove zero-variance features (0.0 = only exact zeros)
    "corr_threshold": 0.95,             # Remove highly correlated features (|r| > threshold)
    "enable_variance_filter": True,     # Enable/disable variance filtering
    "enable_corr_filter": True,         # Enable/disable correlation filtering
}

# 论文超参数（供对比测试）
# Reference: Table S1 from paper
PAPER_GCN_PARAMS = {
    "learning_rate": 0.09,
    "batch_size": 256,
    "dropout": 0.06,
    "hidden_feats": [64, 64, 64],  # 3 layers (num_gnn_layers=3)
    "weight_decay": 0.002,
}
