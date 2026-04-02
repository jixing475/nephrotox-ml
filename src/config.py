import pathlib

# Paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Random seeds for repeated random subsampling (10 iterations)
RANDOM_SEEDS = list(range(10))

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Feature engineering thresholds
VARIANCE_THRESHOLD = 0.0
CORRELATION_THRESHOLD = 0.95
MISSING_VALUE_THRESHOLD = 0.5

# Default configuration key
DEFAULT_CONFIG_KEY = "RF_RDKit"

# Hyperparameter tuning defaults
TUNING_CV_FOLDS = 5
BEST_PARAMS_FILENAME = "best_params.json"
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = None  # seconds, or None for no timeout

# Descriptor sets and corresponding data paths
DESCRIPTOR_SETS = {
    "RDKit": {
        "train_file": INPUT_DIR / "cleaned_data_train_rdkit_desc.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_rdkit_desc.csv",
    },
    "ChemoPy2d": {
        "train_file": INPUT_DIR / "cleaned_data_train_chemopy2d.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_chemopy2d.csv",
    },
    "GraphOnly": {
        "train_file": INPUT_DIR / "cleaned_data_train_graphonly.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_graphonly.csv",
    },
    "KlekotaRoth": {
        "train_file": INPUT_DIR / "cleaned_data_train_klekota_roth.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_klekota_roth.csv",
    },
    "MACCS": {
        "train_file": INPUT_DIR / "cleaned_data_train_maccs.csv",
        "test_file": INPUT_DIR / "cleaned_data_test_maccs.csv",
    },
}


def _build_model_configs(model_class: str, prefix: str):
    """Create MODEL_CONFIGS entries for each descriptor set."""
    return {
        f"{prefix}_{descriptor}": {
            "model_class": model_class,
            "train_file": paths["train_file"],
            "test_file": paths["test_file"],
        }
        for descriptor, paths in DESCRIPTOR_SETS.items()
    }


# Model configurations: 7 model classes × descriptor sets
MODEL_CONFIGS = {
    **_build_model_configs("RandomForestClassifier", "RF"),
    **_build_model_configs("SVC", "SVM"),
    **_build_model_configs("XGBClassifier", "XGB"),
    **_build_model_configs("LGBMClassifier", "LGBM"),
    **_build_model_configs("AdaBoostClassifier", "ADA"),
    **_build_model_configs("QuadraticDiscriminantAnalysis", "QDA"),
    **_build_model_configs("LinearDiscriminantAnalysis", "LDA"),
}

# Hyperparameter search grids for all supported models
PARAM_GRIDS = {
    "RandomForestClassifier": {
        "n_estimators": [500, 1000, 1500, 2000],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced"],
    },
    "SVC": {
        "kernel": ["rbf"],
        "gamma": [0.1, 0.5, 1.0, 2.0],
        "C": [0.1, 1.0, 10.0],
        "class_weight": ["balanced"],
    },
    "XGBClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.3, 0.5],
    },
    "LGBMClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.5, 1.0],
    },
    "QuadraticDiscriminantAnalysis": {
        "reg_param": [0.0, 0.1, 0.3, 0.5],
    },
    "LinearDiscriminantAnalysis": {
        "solver": ["lsqr", "eigen"],
        "shrinkage": ["auto", None, 0.1],
    },
}

# Optuna search spaces (continuous/categorical ranges)
# Format: ("type", min, max) for numeric, ("categorical", [options]) for categorical
# Types: "int", "float", "float_log", "categorical", "int_or_none"
OPTUNA_SEARCH_SPACES = {
    "RandomForestClassifier": {
        "n_estimators": ("int", 100, 2000),
        "max_depth": ("int_or_none", 5, 50),
        "max_features": ("categorical", ["sqrt", "log2"]),
        "class_weight": ("categorical", ["balanced"]),
    },
    "SVC": {
        "C": ("float_log", 0.01, 100.0),
        "gamma": ("float_log", 0.001, 10.0),
        "kernel": ("categorical", ["rbf"]),
        "class_weight": ("categorical", ["balanced"]),
    },
    "XGBClassifier": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 15),
        "learning_rate": ("float_log", 0.001, 1.0),
    },
    "LGBMClassifier": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 15),
        "learning_rate": ("float_log", 0.001, 1.0),
    },
    "AdaBoostClassifier": {
        "n_estimators": ("int", 50, 500),
        "learning_rate": ("float_log", 0.01, 2.0),
    },
    "QuadraticDiscriminantAnalysis": {
        "reg_param": ("float", 0.0, 0.5),
    },
    "LinearDiscriminantAnalysis": {
        "solver": ("categorical", ["lsqr", "eigen"]),
        "shrinkage": ("categorical", ["auto", None, 0.1]),
    },
}
