"""Training loop for DeepChem GCN model."""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix
from tqdm import tqdm
import deepchem as dc
import optuna

from . import config
from .data_loader import load_data, create_deepchem_dataset
from .featurizers import GraphFeaturizer
from .models import create_gcn_model
from .metrics import calculate_metrics, metrics_to_frame
from .callbacks import EarlyStoppingCallback, fit_with_early_stopping
from .tuning import (
    tune_with_optuna,
    tune_dgllife_with_optuna,
    save_tuned_params,
    load_tuned_params,
)
from .utils import (
    ensure_output_dir_with_confirm,
    save_dataframe,
    save_json,
    log_versions,
    update_json,
)
# train_dgllife_single_seed is imported in run_single_seed when needed


def _run_deepchem_training(
    seed: int,
    df: pd.DataFrame,
    cfg: Dict,
    external_df: pd.DataFrame,
    tuned_params: Dict | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str | None = None,
    use_early_stopping: bool = True,
):
    """Run DeepChem training and evaluation for a single seed.
    
    Args:
        seed: Random seed
        df: Training dataframe
        cfg: Model configuration
        external_df: External test dataframe
        tuned_params: Optional hyperparameters
        early_stopping_patience: Patience for early stopping (None = use config default)
        early_stopping_metric: Metric for early stopping (None = use config default)
        use_early_stopping: Whether to use early stopping
        
    Returns:
        Dictionary with metrics, predictions, and other results
    """
    # Split: 80/10/10 with exact fractions (same as ml module)
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, stratify=df["label"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=1.0 / 9.0,  # 10% of total from remaining 90%
        stratify=train_val_df["label"],
        random_state=seed,
    )

    # Create featurizer with parallel processing and caching
    cache_dir = config.OUTPUT_DIR / "feature_cache"
    featurizer = GraphFeaturizer(n_jobs=-1, cache_dir=cache_dir)

    # Create DeepChem datasets
    train_dataset = create_deepchem_dataset(
        train_df["SMILES"].values,
        train_df["label"].values,
        featurizer
    )
    val_dataset = create_deepchem_dataset(
        val_df["SMILES"].values,
        val_df["label"].values,
        featurizer
    )
    test_dataset = create_deepchem_dataset(
        test_df["SMILES"].values,
        test_df["label"].values,
        featurizer
    )
    external_dataset = create_deepchem_dataset(
        external_df["SMILES"].values,
        external_df["label"].values,
        featurizer
    )

    # Note: For graph neural networks, graph features (GraphData) don't need normalization
    # The MolGraphConvFeaturizer already produces normalized graph representations
    # For classification tasks, labels must remain as integers (0, 1)
    # No transformers needed for GCN models
    transformers = []

    # Create and train GCN model
    model_params = tuned_params.copy() if tuned_params else {}
    model = create_gcn_model(n_tasks=1, mode="classification", **model_params)
    
    # Setup early stopping if enabled
    early_stopping = None
    if use_early_stopping:
        patience = early_stopping_patience if early_stopping_patience is not None else config.TRAINING_PARAMS["early_stopping_patience"]
        metric = early_stopping_metric if early_stopping_metric is not None else config.TRAINING_PARAMS["early_stopping_metric"]
        eval_interval = config.TRAINING_PARAMS["eval_interval"]
        save_best = config.TRAINING_PARAMS["save_best_model"]
        
        # Create temporary directory for best model (per seed)
        import tempfile
        temp_dir = pathlib.Path(tempfile.mkdtemp())
        
        early_stopping = EarlyStoppingCallback(
            val_dataset=val_dataset,
            y_val=val_df["label"].values,
            patience=patience,
            metric=metric,
            eval_interval=eval_interval,
            save_dir=temp_dir,
            save_best_model=save_best,
        )
    
    # Train with early stopping
    nb_epoch = config.TRAINING_PARAMS["nb_epoch"]
    model = fit_with_early_stopping(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        y_val=val_df["label"].values,
        nb_epoch=nb_epoch,
        early_stopping=early_stopping,
    )

    # Evaluate with standard metrics
    metrics = [
        dc.metrics.Metric(dc.metrics.roc_auc_score, name="ROC-AUC"),
        dc.metrics.Metric(dc.metrics.accuracy_score, name="Accuracy"),
    ]

    # Get predictions for metrics calculation
    # For classification, predict() returns probabilities
    # Shape: [n_samples, n_classes] for binary classification (2 classes)
    val_proba_raw = model.predict(val_dataset)
    # Extract probability for positive class (class 1)
    if val_proba_raw.shape[1] == 2:
        val_proba = val_proba_raw[:, 1]  # Positive class probability
    else:
        val_proba = val_proba_raw.flatten()
    val_pred = (val_proba >= 0.5).astype(int)
    
    test_proba_raw = model.predict(test_dataset)
    if test_proba_raw.shape[1] == 2:
        test_proba = test_proba_raw[:, 1]
    else:
        test_proba = test_proba_raw.flatten()
    test_pred = (test_proba >= 0.5).astype(int)
    
    ext_proba_raw = model.predict(external_dataset)
    if ext_proba_raw.shape[1] == 2:
        ext_proba = ext_proba_raw[:, 1]
    else:
        ext_proba = ext_proba_raw.flatten()
    ext_pred = (ext_proba >= 0.5).astype(int)

    # Calculate metrics using sklearn (for consistency with ml module)
    # Use original labels (before normalization) for metrics calculation
    val_y_true = val_df["label"].values
    test_y_true = test_df["label"].values
    ext_y_true = external_df["label"].values
    
    val_metrics = calculate_metrics(
        val_y_true, val_pred.flatten(), val_proba.flatten()
    )
    test_metrics = calculate_metrics(
        test_y_true, test_pred.flatten(), test_proba.flatten()
    )
    ext_metrics = calculate_metrics(
        ext_y_true, ext_pred.flatten(), ext_proba.flatten()
    )

    # Use original labels (before normalization) for all calculations
    ext_y_true = external_df["label"].values

    # Predictions dataframe for external test
    preds_df = pd.DataFrame(
        {
            "seed": seed,
            "ID": external_df["ID"].values,
            "y_true": ext_y_true,
            "y_pred": ext_pred.flatten(),
            "y_proba": ext_proba.flatten(),
        }
    )

    # ROC data (external test)
    fpr, tpr, thresholds = roc_curve(ext_y_true, ext_proba.flatten())
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

    # Confusion matrix (external test)
    cm = confusion_matrix(ext_y_true, ext_pred.flatten())
    cm_df = pd.DataFrame(cm, columns=["pred_0", "pred_1"], index=["true_0", "true_1"])

    # Feature importance (not applicable for GCN, return empty DataFrame)
    fi_df = pd.DataFrame(columns=["feature", "importance"])

    warnings: List[dict] = []

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ext_metrics": ext_metrics,
        "preds_df": preds_df,
        "feature_importance": fi_df,
        "roc": roc_df,
        "confusion": cm_df,
        "warnings": warnings,
    }


def run_single_seed(
    seed: int,
    df: pd.DataFrame,
    cfg: Dict,
    external_df: pd.DataFrame,
    tuned_params: Dict | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str | None = None,
    use_early_stopping: bool = True,
    use_class_weights: bool = False,
    optimize_threshold: bool = False,
):
    """Run training and evaluation for a single seed (backend dispatcher).
    
    Args:
        seed: Random seed
        df: Training dataframe
        cfg: Model configuration
        external_df: External test dataframe
        tuned_params: Optional hyperparameters
        early_stopping_patience: Patience for early stopping (None = use config default)
        early_stopping_metric: Metric for early stopping (None = use config default)
        use_early_stopping: Whether to use early stopping
        use_class_weights: Whether to use class-weighted CrossEntropyLoss
        optimize_threshold: Whether to optimize classification threshold
        
    Returns:
        Dictionary with metrics, predictions, and other results
    """
    backend = cfg.get("backend", "deepchem")
    
    if backend == "deepchem":
        return _run_deepchem_training(
            seed, df, cfg, external_df, tuned_params,
            early_stopping_patience, early_stopping_metric, use_early_stopping
        )
    elif backend == "dgllife":
        from .trainer_dgllife import train_dgllife_single_seed
        return train_dgllife_single_seed(
            seed, df, cfg, external_df, tuned_params,
            early_stopping_patience, early_stopping_metric, use_early_stopping,
            use_class_weights=use_class_weights,
            optimize_threshold=optimize_threshold,
        )
    elif backend == "chemprop":
        from .trainer_chemprop import train_chemprop_single_seed
        return train_chemprop_single_seed(
            seed, df, cfg, external_df, tuned_params,
            early_stopping_patience, early_stopping_metric, use_early_stopping,
            use_class_weights=use_class_weights,
            optimize_threshold=optimize_threshold,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_tuning(
    cfg: dict,
    df_train: pd.DataFrame,
    out_dir: pathlib.Path,
    n_trials: int = 20,
    nb_epoch: int = 50,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str | None = None,
    use_early_stopping: bool = True,
):
    """Run hyperparameter tuning on training data and persist best params.
    
    Args:
        cfg: Model configuration dictionary
        df_train: Training dataframe
        out_dir: Output directory
        n_trials: Number of Optuna trials
        nb_epoch: Number of training epochs per trial
        early_stopping_patience: Patience for early stopping (None = use config default)
        early_stopping_metric: Metric for early stopping (None = use config default)
        use_early_stopping: Whether to use early stopping
        
    Returns:
        Tuple of (best_params, study)
    """
    backend = cfg.get("backend", "deepchem")
    
    if backend == "dgllife":
        # DGLlife tuning
        from .trainer_dgllife import featurize_smiles_to_dgl
        
        # Get model type from config
        model_type = cfg.get("algorithm", "GCN")
        
        # Use 80/20 split for tuning (tune_train / tune_val)
        tune_train_df, tune_val_df = train_test_split(
            df_train, test_size=0.2, stratify=df_train["label"], random_state=42
        )
        
        # Featurize SMILES to DGL graphs
        print("Featurizing training molecules for tuning...")
        train_graphs = featurize_smiles_to_dgl(tune_train_df["SMILES"].values.tolist(), model_type=model_type)
        train_valid_indices = [i for i, g in enumerate(train_graphs) if g is not None]
        train_graphs = [train_graphs[i] for i in train_valid_indices]
        train_labels = tune_train_df.iloc[train_valid_indices]["label"].values
        
        print("Featurizing validation molecules for tuning...")
        val_graphs = featurize_smiles_to_dgl(tune_val_df["SMILES"].values.tolist(), model_type=model_type)
        val_valid_indices = [i for i, g in enumerate(val_graphs) if g is not None]
        val_graphs = [val_graphs[i] for i in val_valid_indices]
        val_labels = tune_val_df.iloc[val_valid_indices]["label"].values
        
        # Select search space based on model type
        if model_type == "GCN":
            search_space = config.DGLLIFE_GCN_SEARCH_SPACE
        elif model_type == "GAT":
            search_space = config.DGLLIFE_GAT_SEARCH_SPACE
        elif model_type == "Weave":
            search_space = config.DGLLIFE_WEAVE_SEARCH_SPACE
        elif model_type == "AttentiveFP":
            search_space = config.DGLLIFE_ATTENTIVEFP_SEARCH_SPACE
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Run Optuna optimization
        best_params, study = tune_dgllife_with_optuna(
            train_graphs=train_graphs,
            train_labels=train_labels,
            val_graphs=val_graphs,
            val_labels=val_labels,
            search_space=search_space,
            model_type=model_type,
            n_trials=n_trials,
            nb_epoch=nb_epoch,
            seed=42,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            use_early_stopping=use_early_stopping,
        )
        
        # Convert parameter names for GCN backward compatibility
        # Other models use their native parameter names
        if model_type == "GCN":
            converted_params = {
                "learning_rate": best_params.get("learning_rate", 0.001),
                "batch_size": best_params.get("batch_size", 128),
                "dropout": best_params.get("dropout", 0.0),
                "graph_conv_layers": best_params.get("hidden_feats", [64, 64, 64]),
                "dense_layer_size": best_params.get("classifier_hidden_feats", 128),
                "weight_decay": best_params.get("weight_decay", 0.0),
            }
        else:
            # For other models, use best_params directly
            converted_params = best_params.copy()
        
        # Save best parameters
        save_tuned_params(converted_params, out_dir / config.BEST_PARAMS_FILENAME)
        
        # Save study trials as JSON
        trials_data = []
        for trial in study.trials:
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            })
        save_json(trials_data, out_dir / "optuna_study.json")
        
        # Save parameter importance if available
        try:
            importance = optuna.importance.get_param_importances(study)
            importance_df = pd.DataFrame(
                list(importance.items()), columns=["parameter", "importance"]
            ).sort_values("importance", ascending=False)
            save_dataframe(importance_df, out_dir / "optuna_importance.csv")
        except Exception:
            # Parameter importance requires at least 2 completed trials
            pass
        
        return converted_params, study
    
    elif backend == "chemprop":
        # Chemprop tuning
        from .trainer_chemprop import _df_to_chemprop_datapoints
        from .tuning import tune_chemprop_with_optuna
        
        # Use 80/20 split for tuning (tune_train / tune_val)
        tune_train_df, tune_val_df = train_test_split(
            df_train, test_size=0.2, stratify=df_train["label"], random_state=42
        )
        
        # Convert DataFrames to chemprop MoleculeDatapoint
        print("Converting training molecules to MoleculeDatapoint for tuning...")
        train_datapoints = _df_to_chemprop_datapoints(tune_train_df)
        
        print("Converting validation molecules to MoleculeDatapoint for tuning...")
        val_datapoints = _df_to_chemprop_datapoints(tune_val_df)
        val_labels = tune_val_df["label"].values
        
        # Run Optuna optimization
        search_space = config.CHEMPROP_DMPNN_SEARCH_SPACE
        best_params, study = tune_chemprop_with_optuna(
            train_datapoints=train_datapoints,
            val_datapoints=val_datapoints,
            val_labels=val_labels,
            search_space=search_space,
            n_trials=n_trials,
            nb_epoch=nb_epoch,
            seed=42,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            use_early_stopping=use_early_stopping,
        )
        
        # Save best parameters
        save_tuned_params(best_params, out_dir / config.BEST_PARAMS_FILENAME)
        
        # Save study trials as JSON
        trials_data = []
        for trial in study.trials:
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            })
        save_json(trials_data, out_dir / "optuna_study.json")
        
        # Save parameter importance if available
        try:
            importance = optuna.importance.get_param_importances(study)
            importance_df = pd.DataFrame(
                list(importance.items()), columns=["parameter", "importance"]
            ).sort_values("importance", ascending=False)
            save_dataframe(importance_df, out_dir / "optuna_importance.csv")
        except Exception:
            # Parameter importance requires at least 2 completed trials
            pass
        
        return best_params, study
    
    # DeepChem tuning (existing logic)
    # Use 80/20 split for tuning (tune_train / tune_val)
    tune_train_df, tune_val_df = train_test_split(
        df_train, test_size=0.2, stratify=df_train["label"], random_state=42
    )
    
    # Create featurizer with parallel processing and caching
    cache_dir = config.OUTPUT_DIR / "feature_cache"
    featurizer = GraphFeaturizer(n_jobs=-1, cache_dir=cache_dir)
    
    # Create DeepChem datasets
    tune_train_dataset = create_deepchem_dataset(
        tune_train_df["SMILES"].values,
        tune_train_df["label"].values,
        featurizer
    )
    tune_val_dataset = create_deepchem_dataset(
        tune_val_df["SMILES"].values,
        tune_val_df["label"].values,
        featurizer
    )
    
    # Get validation labels for metric calculation
    y_val = tune_val_df["label"].values
    
    # Run Optuna optimization
    search_space = config.GCN_SEARCH_SPACE
    best_params, study = tune_with_optuna(
        tune_train_dataset,
        tune_val_dataset,
        y_val,
        search_space,
        n_trials=n_trials,
        nb_epoch=nb_epoch,
        seed=42,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        use_early_stopping=use_early_stopping,
    )
    
    # Save best parameters
    save_tuned_params(best_params, out_dir / config.BEST_PARAMS_FILENAME)
    
    # Save study trials as JSON
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": trial.state.name,
        })
    save_json(trials_data, out_dir / "optuna_study.json")
    
    # Save parameter importance if available
    try:
        importance = optuna.importance.get_param_importances(study)
        importance_df = pd.DataFrame(
            list(importance.items()), columns=["parameter", "importance"]
        ).sort_values("importance", ascending=False)
        save_dataframe(importance_df, out_dir / "optuna_importance.csv")
    except Exception:
        # Parameter importance requires at least 2 completed trials
        pass
    
    return best_params, study


def train_final(
    seed: int,
    df_train: pd.DataFrame,
    cfg: Dict,
    external_df: pd.DataFrame,
    out_dir: pathlib.Path,
    config_key: str,
    tuned_params: Dict | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str | None = None,
    use_early_stopping: bool = True,
):
    """Train a single final model on all training data and save bundle.
    
    Args:
        seed: Random seed
        df_train: Training dataframe
        cfg: Model configuration dictionary
        external_df: External test dataframe
        out_dir: Output directory
        config_key: Configuration key (e.g., "GCN_Graph")
        tuned_params: Optional hyperparameters
        early_stopping_patience: Patience for early stopping (None = use config default)
        early_stopping_metric: Metric for early stopping (None = use config default)
        use_early_stopping: Whether to use early stopping
        
    Returns:
        Tuple of (model_path, ext_metrics)
    """
    # Create featurizer with parallel processing and caching
    cache_dir = config.OUTPUT_DIR / "feature_cache"
    featurizer = GraphFeaturizer(n_jobs=-1, cache_dir=cache_dir)
    
    # Create DeepChem dataset from all training data
    train_dataset = create_deepchem_dataset(
        df_train["SMILES"].values,
        df_train["label"].values,
        featurizer
    )
    external_dataset = create_deepchem_dataset(
        external_df["SMILES"].values,
        external_df["label"].values,
        featurizer
    )
    
    # For final training, we need a validation set for early stopping
    # Split training data: 90% train, 10% val for early stopping
    from sklearn.model_selection import train_test_split
    train_val_df, val_df = train_test_split(
        df_train, test_size=0.1, stratify=df_train["label"], random_state=seed
    )
    
    train_final_dataset = create_deepchem_dataset(
        train_val_df["SMILES"].values,
        train_val_df["label"].values,
        featurizer
    )
    val_final_dataset = create_deepchem_dataset(
        val_df["SMILES"].values,
        val_df["label"].values,
        featurizer
    )
    
    # Create and train GCN model
    model_params = tuned_params.copy() if tuned_params else {}
    model = create_gcn_model(n_tasks=1, mode="classification", **model_params)
    
    # Setup early stopping if enabled
    early_stopping = None
    if use_early_stopping:
        patience = early_stopping_patience if early_stopping_patience is not None else config.TRAINING_PARAMS["early_stopping_patience"]
        metric = early_stopping_metric if early_stopping_metric is not None else config.TRAINING_PARAMS["early_stopping_metric"]
        eval_interval = config.TRAINING_PARAMS["eval_interval"]
        save_best = config.TRAINING_PARAMS["save_best_model"]
        
        # Use output directory for best model
        best_model_dir = out_dir / "best_model_temp"
        
        early_stopping = EarlyStoppingCallback(
            val_dataset=val_final_dataset,
            y_val=val_df["label"].values,
            patience=patience,
            metric=metric,
            eval_interval=eval_interval,
            save_dir=best_model_dir,
            save_best_model=save_best,
        )
    
    # Train with early stopping
    nb_epoch = config.TRAINING_PARAMS["nb_epoch"]
    model = fit_with_early_stopping(
        model=model,
        train_dataset=train_final_dataset,
        val_dataset=val_final_dataset,
        y_val=val_df["label"].values,
        nb_epoch=nb_epoch,
        early_stopping=early_stopping,
    )
    
    # External evaluation
    ext_proba_raw = model.predict(external_dataset)
    if ext_proba_raw.shape[1] == 2:
        ext_proba = ext_proba_raw[:, 1]
    else:
        ext_proba = ext_proba_raw.flatten()
    ext_pred = (ext_proba >= 0.5).astype(int)
    ext_y_true = external_df["label"].values
    ext_metrics = calculate_metrics(ext_y_true, ext_pred.flatten(), ext_proba.flatten())
    
    # Save model (DeepChem uses model.save() / model.restore())
    model_dir = out_dir / f"{config_key.lower()}_final_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir)
    
    # Save featurizer info and params as JSON (featurizer is stateless, just save class name)
    bundle_info = {
        "featurizer": "GraphFeaturizer",
        "params": tuned_params.copy() if tuned_params else {},
        "seed": seed,
        "model_dir": str(model_dir),
    }
    save_json(bundle_info, out_dir / f"{config_key.lower()}_final_info.json")
    
    # Save metrics and predictions
    save_json({"external": ext_metrics}, out_dir / "final_metrics.json")
    preds_df = pd.DataFrame(
        {
            "ID": external_df["ID"].values,
            "y_true": ext_y_true,
            "y_pred": ext_pred.flatten(),
            "y_proba": ext_proba.flatten(),
        }
    )
    save_dataframe(preds_df, out_dir / "final_predictions.csv")
    
    return model_dir, ext_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=config.DEFAULT_CONFIG_KEY,
        help="Config key from MODEL_CONFIGS"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning with Optuna"
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=config.OPTUNA_N_TRIALS,
        help="Number of Optuna trials (only used with --tune)"
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Number of random seeds to use (default: all 10 seeds)"
    )
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Train once on full train data and save final model",
    )
    parser.add_argument(
        "--final-seed",
        type=int,
        default=0,
        help="Seed used when --save-final is set"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (overrides config default)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["roc_auc", "loss"],
        default=None,
        help="Early stopping metric: 'roc_auc' or 'loss' (overrides config default)"
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping"
    )
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use class-weighted CrossEntropyLoss for imbalanced data"
    )
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Optimize classification threshold using Youden's J on validation set"
    )
    args = parser.parse_args()

    cfg = config.MODEL_CONFIGS.get(args.config)
    if cfg is None:
        raise ValueError(f"Config key {args.config} not found.")

    train_path = pathlib.Path(cfg["train_file"])
    test_path = pathlib.Path(cfg["test_file"])
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    df_train = load_data(train_path)
    df_external = load_data(test_path)

    # Ensure output directory exists (files will be overwritten)
    out_dir = ensure_output_dir_with_confirm(config.OUTPUT_DIR, args.config)
    tuned_params_path = out_dir / config.BEST_PARAMS_FILENAME

    # Tuning phase only
    if args.tune:
        tuned_params, study = run_tuning(
            cfg, df_train, out_dir, n_trials=args.optuna_trials, nb_epoch=config.TRAINING_PARAMS["nb_epoch"],
            early_stopping_patience=args.patience,
            early_stopping_metric=args.metric,
            use_early_stopping=not args.no_early_stop,
        )
        print(f"Tuning complete. Best params saved to: {tuned_params_path}")
        print("Best params:", tuned_params)
        return

    # Load tuned params if available
    tuned_params = None
    if tuned_params_path.exists():
        tuned_params = load_tuned_params(tuned_params_path)
        print(f"Loaded tuned params from: {tuned_params_path}")

    # If only saving a single final model
    if args.save_final:
        model_path, ext_metrics = train_final(
            args.final_seed, df_train, cfg, df_external, out_dir, args.config, tuned_params,
            early_stopping_patience=args.patience,
            early_stopping_metric=args.metric,
            use_early_stopping=not args.no_early_stop,
        )
        print(f"Saved final model to: {model_path}")
        print("External metrics:", ext_metrics)
        log_versions(out_dir / "experiment_info.json")
        paper_targets = {
            "val": {"AUC": 91.3, "ACC": 82.4, "SE": 78.9, "F1": 81.8, "Kappa": 64.7},
            "test": {"AUC": 91.7, "ACC": 83.1, "SE": 78.4, "F1": 82.4, "Kappa": 66.2},
        }
        save_json(paper_targets, out_dir / "paper_targets.json")
        return

    # Determine which seeds to use
    seeds_to_use = config.RANDOM_SEEDS
    if args.n_seeds is not None:
        seeds_to_use = config.RANDOM_SEEDS[:args.n_seeds]
        print(f"Using {len(seeds_to_use)} seed(s) for testing: {seeds_to_use}")

    # Containers for multi-seed experiment
    val_metrics_list: List[dict] = []
    test_metrics_list: List[dict] = []
    ext_metrics_list: List[dict] = []
    preds_all: List[pd.DataFrame] = []
    roc_all: List[pd.DataFrame] = []
    fi_all: List[pd.DataFrame] = []
    cm_all: List[pd.DataFrame] = []
    run_warnings: List[dict] = []

    # Run multi-seed evaluation
    for seed in tqdm(seeds_to_use, desc="Seeds"):
        results = run_single_seed(
            seed, df_train, cfg, df_external, tuned_params=tuned_params,
            early_stopping_patience=args.patience,
            early_stopping_metric=args.metric,
            use_early_stopping=not args.no_early_stop,
            use_class_weights=args.class_weights,
            optimize_threshold=args.optimize_threshold,
        )
        val_metrics_list.append(results["val_metrics"])
        test_metrics_list.append(results["test_metrics"])
        ext_metrics_list.append(results["ext_metrics"])
        preds_all.append(results["preds_df"])
        roc_all.append(results["roc"].assign(seed=seed))
        fi_all.append(results["feature_importance"].assign(seed=seed))
        cm_all.append(results["confusion"].assign(seed=seed))
        run_warnings.extend(results.get("warnings", []))

    # Aggregate metrics
    val_df = metrics_to_frame(val_metrics_list, "val", seeds_to_use)
    test_df = metrics_to_frame(test_metrics_list, "test", seeds_to_use)
    ext_df = metrics_to_frame(ext_metrics_list, "external", seeds_to_use)

    # Mean/std summary
    def summarize(df, set_name):
        metric_cols = [c for c in df.columns if c not in ["seed", "set"]]
        summary = df[metric_cols].agg(["mean", "std"]).T.reset_index()
        summary.columns = ["metric", "mean", "std"]
        summary["set"] = set_name
        return summary

    summary_df = pd.concat([
        summarize(val_df, "val"),
        summarize(test_df, "test"),
        summarize(ext_df, "external"),
    ], ignore_index=True)

    # Save results
    per_fold_df = pd.concat([val_df, test_df, ext_df], ignore_index=True)
    save_dataframe(per_fold_df, out_dir / "cv_results_per_fold.csv")
    save_dataframe(summary_df, out_dir / "cv_summary.csv")

    # Predictions and curves
    preds_concat = pd.concat(preds_all, ignore_index=True)
    preds_concat["set"] = "external"
    save_dataframe(preds_concat, out_dir / "test_predictions.csv")

    save_dataframe(pd.concat(roc_all, ignore_index=True), out_dir / "roc_curve_data.csv")
    save_dataframe(pd.concat(fi_all, ignore_index=True), out_dir / "feature_importance.csv")
    # Confusion matrices per seed
    cm_concat = pd.concat(cm_all, keys=seeds_to_use, names=["seed", "true"])
    cm_concat.to_csv(out_dir / "confusion_matrix.csv")

    # Experiment info / versions
    log_versions(out_dir / "experiment_info.json")
    if run_warnings:
        update_json(out_dir / "experiment_info.json", {"warnings": run_warnings})

    # Metrics comparison with paper (if desired)
    paper_targets = {
        "val": {"AUC": 91.3, "ACC": 82.4, "SE": 78.9, "F1": 81.8, "Kappa": 64.7},
        "test": {"AUC": 91.7, "ACC": 83.1, "SE": 78.4, "F1": 82.4, "Kappa": 66.2},
    }
    save_json(paper_targets, out_dir / "paper_targets.json")

    print(f"Training complete! Results saved to: {out_dir}")


if __name__ == "__main__":
    main()

