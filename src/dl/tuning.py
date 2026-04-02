"""Hyperparameter tuning for DeepChem GCN model using Optuna."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import deepchem as dc
import torch

from .models import create_gcn_model
from .data_loader import create_deepchem_dataset
from .featurizers import GraphFeaturizer
from .callbacks import EarlyStoppingCallback, fit_with_early_stopping
from .models_dgllife import (
    create_dgllife_gcn_model,
    create_dgllife_gat_model,
    create_dgllife_weave_model,
    create_dgllife_attentivefp_model,
)
from .trainer_dgllife import DGLGraphDataset, collate_graphs
from . import config


def get_available_gpus() -> list[int]:
    """Get list of available GPU device IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def set_gpu_device(gpu_id: int) -> None:
    """Set CUDA_VISIBLE_DEVICES to use specific GPU.
    
    Args:
        gpu_id: GPU device ID to use
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Force PyTorch to re-initialize CUDA with new device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, device 0 is the selected GPU


def save_tuned_params(params: dict, path: Path) -> None:
    """Save tuned hyperparameters to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def load_tuned_params(path: Path) -> dict:
    """Load tuned hyperparameters from JSON file."""
    with open(path) as f:
        return json.load(f)


def _sample_params(trial: optuna.Trial, search_space: Dict[str, Tuple[str, Any]]) -> Dict[str, Any]:
    """Sample hyperparameters from search space using Optuna trial.
    
    Args:
        trial: Optuna trial object
        search_space: Dictionary mapping parameter names to (type, ...) tuples
            Types: "int", "float", "float_log", "categorical"
            
    Returns:
        Dictionary of sampled parameters
    """
    params = {}
    for name, spec in search_space.items():
        spec_type = spec[0]
        
        if spec_type == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif spec_type == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif spec_type == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif spec_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown search space type: {spec_type}")
    
    return params


def tune_with_optuna(
    train_dataset: dc.data.Dataset,
    val_dataset: dc.data.Dataset,
    y_val: Any,
    search_space: Dict[str, Tuple[str, Any]],
    n_trials: int = 20,
    nb_epoch: int = 50,
    seed: int = 42,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str | None = None,
    use_early_stopping: bool = True,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    gpu_id: Optional[int] = None,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Run Optuna optimization for GCN model.
    
    Note: DeepChem GCNModel does not support sklearn's cross_val_score,
    so we use a single train/val split for tuning evaluation.
    
    Args:
        train_dataset: DeepChem training dataset
        val_dataset: DeepChem validation dataset
        y_val: True labels for validation set (for metric calculation)
        search_space: Dictionary mapping parameter names to (type, ...) tuples
        n_trials: Number of Optuna trials
        nb_epoch: Number of training epochs per trial
        seed: Random seed for Optuna sampler
        early_stopping_patience: Patience for early stopping (None = use config default)
        early_stopping_metric: Metric for early stopping (None = use config default)
        use_early_stopping: Whether to use early stopping
        storage: Optuna storage URL (e.g., "sqlite:///study.db") for distributed optimization
        study_name: Name of the study (required if storage is provided)
        gpu_id: GPU device ID to use (None = use default)
        
    Returns:
        Tuple of (best_params, study)
    """
    # Set GPU device if specified
    if gpu_id is not None:
        set_gpu_device(gpu_id)
    
    def objective(trial):
        # Sample hyperparameters from search space
        params = _sample_params(trial, search_space)
        
        # Create GCN model with sampled parameters
        model = create_gcn_model(n_tasks=1, mode="classification", **params)
        
        # Setup early stopping if enabled
        early_stopping = None
        if use_early_stopping:
            patience = early_stopping_patience if early_stopping_patience is not None else config.TRAINING_PARAMS["early_stopping_patience"]
            metric = early_stopping_metric if early_stopping_metric is not None else config.TRAINING_PARAMS["early_stopping_metric"]
            eval_interval = config.TRAINING_PARAMS["eval_interval"]
            
            # Don't save best model during tuning (we just need the score)
            early_stopping = EarlyStoppingCallback(
                val_dataset=val_dataset,
                y_val=y_val,
                patience=patience,
                metric=metric,
                eval_interval=eval_interval,
                save_dir=None,
                save_best_model=False,
            )
        
        # Train model with early stopping
        model = fit_with_early_stopping(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            y_val=y_val,
            nb_epoch=nb_epoch,
            early_stopping=early_stopping,
        )
        
        # Evaluate on validation set
        val_proba_raw = model.predict(val_dataset)

        # Handle different prediction formats from DeepChem
        if hasattr(val_proba_raw, 'shape') and len(val_proba_raw.shape) == 3:
            # Shape: (n_samples, 1, 2) - extract positive class probabilities
            val_proba = val_proba_raw[:, 0, 1]
        elif hasattr(val_proba_raw, 'shape') and len(val_proba_raw.shape) == 2 and val_proba_raw.shape[1] == 2:
            # Shape: (n_samples, 2) - extract positive class probabilities
            val_proba = val_proba_raw[:, 1]
        else:
            # Fallback: flatten and ensure we have probabilities
            val_proba = np.array(val_proba_raw).flatten()
        
        # Calculate ROC-AUC score
        score = roc_auc_score(y_val, val_proba)
        
        # Clean up model to free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return score
    
    # Create or load Optuna study
    if storage is not None:
        # Use persistent storage for distributed optimization
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            load_if_exists=True,  # Allow multiple workers to share the study
        )
    else:
        # In-memory study for single-process optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Return best parameters and study
    return study.best_params, study


def run_worker(
    gpu_id: int,
    storage: str,
    study_name: str,
    config_key: str,
    train_csv: Path,
    n_trials_per_worker: int,
    nb_epoch: int,
    early_stopping_patience: int | None,
    early_stopping_metric: str | None,
    use_early_stopping: bool,
    seed: int,
    n_cpu_per_worker: int = -1,
) -> None:
    """Run a single Optuna worker on a specific GPU.
    
    This function is designed to be called in a subprocess.
    
    Args:
        gpu_id: GPU device ID to use
        storage: Optuna storage URL
        study_name: Name of the shared study
        config_key: Configuration key (e.g., "DMPNN_Chemprop_Graph")
        train_csv: Path to training CSV file
        n_trials_per_worker: Number of trials this worker should run
        nb_epoch: Number of training epochs per trial
        early_stopping_patience: Patience for early stopping
        early_stopping_metric: Metric for early stopping
        use_early_stopping: Whether to use early stopping
        seed: Random seed (will be modified by gpu_id for diversity)
        n_cpu_per_worker: Number of CPUs for parallel featurization per worker
    """
    from sklearn.model_selection import train_test_split
    from .data_loader import load_data
    
    # Set GPU device BEFORE importing any CUDA-dependent code
    set_gpu_device(gpu_id)
    
    print(f"[GPU {gpu_id}] Worker starting, assigned {n_trials_per_worker} trials")
    print(f"[GPU {gpu_id}] Config: {config_key}")
    
    # Get configuration
    cfg = config.MODEL_CONFIGS.get(config_key)
    if cfg is None:
        raise ValueError(f"Config key {config_key} not found.")
    
    backend = cfg.get("backend", "deepchem")
    algorithm = cfg.get("algorithm", "GCN")
    
    # Load and prepare data
    df_train = load_data(train_csv)
    
    # Use 80/20 split for tuning (same as run_tuning)
    tune_train_df, tune_val_df = train_test_split(
        df_train, test_size=0.2, stratify=df_train["label"], random_state=42
    )
    
    worker_seed = seed + gpu_id  # Different seed per worker for diversity
    
    # Dispatch to appropriate backend
    if backend == "deepchem":
        # DeepChem GCN tuning
        cache_dir = Path(train_csv).parent.parent / "output" / "feature_cache"
        featurizer = GraphFeaturizer(n_jobs=n_cpu_per_worker, cache_dir=cache_dir)
        
        print(f"[GPU {gpu_id}] Featurizing training data...")
        tune_train_dataset = create_deepchem_dataset(
            tune_train_df["SMILES"].values,
            tune_train_df["label"].values,
            featurizer
        )
        print(f"[GPU {gpu_id}] Featurizing validation data...")
        tune_val_dataset = create_deepchem_dataset(
            tune_val_df["SMILES"].values,
            tune_val_df["label"].values,
            featurizer
        )
        y_val = tune_val_df["label"].values
        
        print(f"[GPU {gpu_id}] Starting Optuna optimization (DeepChem GCN)...")
        
        search_space = config.GCN_SEARCH_SPACE
        tune_with_optuna(
            tune_train_dataset,
            tune_val_dataset,
            y_val,
            search_space,
            n_trials=n_trials_per_worker,
            nb_epoch=nb_epoch,
            seed=worker_seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            use_early_stopping=use_early_stopping,
            storage=storage,
            study_name=study_name,
            gpu_id=None,  # Already set at process start
        )
    
    elif backend == "dgllife":
        # DGLlife tuning
        from .trainer_dgllife import featurize_smiles_to_dgl
        
        print(f"[GPU {gpu_id}] Featurizing training molecules (DGLlife {algorithm})...")
        train_graphs = featurize_smiles_to_dgl(tune_train_df["SMILES"].values.tolist(), model_type=algorithm)
        train_valid_indices = [i for i, g in enumerate(train_graphs) if g is not None]
        train_graphs = [train_graphs[i] for i in train_valid_indices]
        train_labels = tune_train_df.iloc[train_valid_indices]["label"].values
        
        print(f"[GPU {gpu_id}] Featurizing validation molecules...")
        val_graphs = featurize_smiles_to_dgl(tune_val_df["SMILES"].values.tolist(), model_type=algorithm)
        val_valid_indices = [i for i, g in enumerate(val_graphs) if g is not None]
        val_graphs = [val_graphs[i] for i in val_valid_indices]
        val_labels = tune_val_df.iloc[val_valid_indices]["label"].values
        
        # Select search space based on model type
        if algorithm == "GCN":
            search_space = config.DGLLIFE_GCN_SEARCH_SPACE
        elif algorithm == "GAT":
            search_space = config.DGLLIFE_GAT_SEARCH_SPACE
        elif algorithm == "Weave":
            search_space = config.DGLLIFE_WEAVE_SEARCH_SPACE
        elif algorithm == "AttentiveFP":
            search_space = config.DGLLIFE_ATTENTIVEFP_SEARCH_SPACE
        else:
            raise ValueError(f"Unknown DGLlife model type: {algorithm}")
        
        print(f"[GPU {gpu_id}] Starting Optuna optimization (DGLlife {algorithm})...")
        
        tune_dgllife_with_optuna(
            train_graphs=train_graphs,
            train_labels=train_labels,
            val_graphs=val_graphs,
            val_labels=val_labels,
            search_space=search_space,
            model_type=algorithm,
            n_trials=n_trials_per_worker,
            nb_epoch=nb_epoch,
            seed=worker_seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            use_early_stopping=use_early_stopping,
            storage=storage,
            study_name=study_name,
            gpu_id=None,  # Already set at process start
        )
    
    elif backend == "chemprop":
        # Chemprop tuning
        from .trainer_chemprop import _df_to_chemprop_datapoints
        
        print(f"[GPU {gpu_id}] Converting training molecules to MoleculeDatapoint (Chemprop D-MPNN)...")
        train_datapoints = _df_to_chemprop_datapoints(tune_train_df)
        
        print(f"[GPU {gpu_id}] Converting validation molecules to MoleculeDatapoint...")
        val_datapoints = _df_to_chemprop_datapoints(tune_val_df)
        val_labels = tune_val_df["label"].values
        
        print(f"[GPU {gpu_id}] Starting Optuna optimization (Chemprop D-MPNN)...")
        
        search_space = config.CHEMPROP_DMPNN_SEARCH_SPACE
        tune_chemprop_with_optuna(
            train_datapoints=train_datapoints,
            val_datapoints=val_datapoints,
            val_labels=val_labels,
            search_space=search_space,
            n_trials=n_trials_per_worker,
            nb_epoch=nb_epoch,
            seed=worker_seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            use_early_stopping=use_early_stopping,
            storage=storage,
            study_name=study_name,
            gpu_id=None,  # Already set at process start
        )
    
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    print(f"[GPU {gpu_id}] Worker finished")


def tune_dgllife_with_optuna(
    train_graphs: list,
    train_labels: np.ndarray,
    val_graphs: list,
    val_labels: np.ndarray,
    search_space: Dict[str, Tuple[str, Any]],
    model_type: str = "GCN",
    n_trials: int = 20,
    nb_epoch: int = 50,
    seed: int = 42,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str | None = None,
    use_early_stopping: bool = True,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    gpu_id: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Run Optuna optimization for DGLlife model (GCN, GAT, Weave, or AttentiveFP).
    
    Args:
        train_graphs: List of DGL graphs for training
        train_labels: Training labels array
        val_graphs: List of DGL graphs for validation
        val_labels: Validation labels array
        search_space: Dictionary mapping parameter names to (type, ...) tuples
        model_type: Model type ("GCN", "GAT", "Weave", "AttentiveFP")
        n_trials: Number of Optuna trials
        nb_epoch: Number of training epochs per trial
        seed: Random seed for Optuna sampler
        early_stopping_patience: Patience for early stopping (None = use config default)
        early_stopping_metric: Metric for early stopping (None = use config default)
        use_early_stopping: Whether to use early stopping
        storage: Optuna storage URL (e.g., "sqlite:///study.db") for distributed optimization
        study_name: Name of the study (required if storage is provided)
        gpu_id: GPU device ID to use (None = use default)
        device: Device string ("cuda" or "cpu", None = auto-detect)
        
    Returns:
        Tuple of (best_params, study)
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    
    # Set GPU device if specified
    if gpu_id is not None:
        set_gpu_device(gpu_id)
    
    # Set device
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_obj = torch.device(device_str)
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create datasets
    train_dataset = DGLGraphDataset(train_graphs, train_labels)
    val_dataset = DGLGraphDataset(val_graphs, val_labels)
    
    # Determine if model uses edge features
    uses_edge_features = model_type in ["Weave", "AttentiveFP"]
    
    def objective(trial):
        # Sample hyperparameters from search space
        params = _sample_params(trial, search_space)
        
        # Extract common parameters
        batch_size = params.pop("batch_size", 128)
        learning_rate = params.pop("learning_rate", 0.001)
        weight_decay = params.pop("weight_decay", 0.0)
        dropout = params.pop("dropout", 0.0)
        
        # Create model based on type
        if model_type == "GCN":
            hidden_feats = params.pop("hidden_feats", [64, 64, 64])
            classifier_hidden_feats = params.pop("classifier_hidden_feats", 128)
            model = create_dgllife_gcn_model(
                in_feats=74,  # CanonicalAtomFeaturizer produces 74D features
                hidden_feats=hidden_feats,
                classifier_hidden_feats=classifier_hidden_feats,
                dropout=dropout,
                pooling="sum",
            ).to(device_obj)
        elif model_type == "GAT":
            hidden_feats = params.pop("hidden_feats", [64, 64, 64])
            num_heads = params.pop("num_heads", [8, 8, 8])
            feat_drops = params.pop("feat_drops", [dropout] * len(hidden_feats))
            attn_drops = params.pop("attn_drops", [dropout] * len(hidden_feats))
            classifier_hidden_feats = params.pop("classifier_hidden_feats", 128)
            predictor_hidden_feats = params.pop("predictor_hidden_feats", 128)
            model = create_dgllife_gat_model(
                in_feats=74,  # CanonicalAtomFeaturizer produces 74D features
                hidden_feats=hidden_feats,
                num_heads=num_heads,
                feat_drops=feat_drops,
                attn_drops=attn_drops,
                classifier_hidden_feats=classifier_hidden_feats,
                classifier_dropout=dropout,
                predictor_hidden_feats=predictor_hidden_feats,
                predictor_dropout=dropout,
            ).to(device_obj)
        elif model_type == "Weave":
            num_gnn_layers = params.pop("num_gnn_layers", 2)
            gnn_hidden_feats = params.pop("gnn_hidden_feats", 50)
            graph_feats = params.pop("graph_feats", 128)
            model = create_dgllife_weave_model(
                node_in_feats=27,  # WeaveAtomFeaturizer produces 27D features
                edge_in_feats=12,  # WeaveEdgeFeaturizer produces 12D features
                num_gnn_layers=num_gnn_layers,
                gnn_hidden_feats=gnn_hidden_feats,
                graph_feats=graph_feats,
            ).to(device_obj)
        elif model_type == "AttentiveFP":
            num_layers = params.pop("num_layers", 2)
            num_timesteps = params.pop("num_timesteps", 2)
            graph_feat_size = params.pop("graph_feat_size", 200)
            model = create_dgllife_attentivefp_model(
                node_feat_size=39,  # AttentiveFPAtomFeaturizer produces 39D features
                edge_feat_size=10,  # AttentiveFPBondFeaturizer produces 10D features
                num_layers=num_layers,
                num_timesteps=num_timesteps,
                graph_feat_size=graph_feat_size,
                dropout=dropout,
            ).to(device_obj)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_graphs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
        )
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training parameters
        patience = early_stopping_patience if early_stopping_patience is not None else config.TRAINING_PARAMS["early_stopping_patience"]
        metric = early_stopping_metric if early_stopping_metric is not None else config.TRAINING_PARAMS["early_stopping_metric"]
        eval_interval = config.TRAINING_PARAMS["eval_interval"]
        maximize = metric == "roc_auc"
        
        best_val_score = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(nb_epoch):
            # Training phase
            model.train()
            for bg, labels in train_loader:
                bg = bg.to(device_obj)
                labels = labels.to(device_obj)
                
                # Forward pass (different for different model types)
                if model_type == "GCN":
                    logits = model(bg, bg.ndata["h"])
                elif model_type == "GAT":
                    logits = model(bg, bg.ndata["h"])
                elif model_type == "Weave":
                    logits = model(bg, bg.ndata["h"], bg.edata["e"])
                elif model_type == "AttentiveFP":
                    logits = model(bg, bg.ndata["h"], bg.edata["e"])
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                loss = criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation phase (at eval_interval or last epoch)
            if (epoch + 1) % eval_interval == 0 or epoch == nb_epoch - 1:
                model.eval()
                val_proba_list = []
                val_labels_list = []
                
                with torch.no_grad():
                    for bg, labels in val_loader:
                        bg = bg.to(device_obj)
                        if model_type == "GCN":
                            logits = model(bg, bg.ndata["h"])
                        elif model_type == "GAT":
                            logits = model(bg, bg.ndata["h"])
                        elif model_type == "Weave":
                            logits = model(bg, bg.ndata["h"], bg.edata["e"])
                        elif model_type == "AttentiveFP":
                            logits = model(bg, bg.ndata["h"], bg.edata["e"])
                        else:
                            raise ValueError(f"Unknown model type: {model_type}")
                        proba = torch.softmax(logits, dim=1)
                        val_proba_list.append(proba.cpu().numpy())
                        val_labels_list.append(labels.numpy())
                
                val_proba = np.concatenate(val_proba_list)[:, 1]  # Positive class probability
                val_labels_array = np.concatenate(val_labels_list)
                
                # Calculate metric
                if metric == "roc_auc":
                    val_score = roc_auc_score(val_labels_array, val_proba)
                elif metric == "loss":
                    epsilon = 1e-15
                    val_proba_clipped = np.clip(val_proba, epsilon, 1 - epsilon)
                    loss_val = -np.mean(
                        val_labels_array * np.log(val_proba_clipped) +
                        (1 - val_labels_array) * np.log(1 - val_proba_clipped)
                    )
                    val_score = -loss_val  # Negate for consistent comparison
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                # Check for improvement
                is_better = False
                if best_val_score is None:
                    is_better = True
                elif maximize:
                    is_better = val_score > best_val_score
                else:
                    is_better = val_score > best_val_score
                
                if is_better:
                    best_val_score = val_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience and use_early_stopping:
                        break
        
        # Report best score to Optuna
        score = best_val_score if best_val_score is not None else 0.0
        
        # Clean up model to free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return score
    
    # Create or load Optuna study
    if storage is not None:
        # Use persistent storage for distributed optimization
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            load_if_exists=True,  # Allow multiple workers to share the study
        )
    else:
        # In-memory study for single-process optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Return best parameters and study
    return study.best_params, study


def tune_chemprop_with_optuna(
    train_datapoints: list,
    val_datapoints: list,
    val_labels: np.ndarray,
    search_space: Dict[str, Tuple[str, Any]],
    n_trials: int = 20,
    nb_epoch: int = 50,
    seed: int = 42,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str | None = None,
    use_early_stopping: bool = True,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    gpu_id: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Run Optuna optimization for Chemprop D-MPNN model.
    
    Args:
        train_datapoints: List of chemprop MoleculeDatapoint for training
        val_datapoints: List of chemprop MoleculeDatapoint for validation
        val_labels: Validation labels array
        search_space: Dictionary mapping parameter names to (type, ...) tuples
        n_trials: Number of Optuna trials
        nb_epoch: Number of training epochs per trial
        seed: Random seed for Optuna sampler
        early_stopping_patience: Patience for early stopping (None = use config default)
        early_stopping_metric: Metric for early stopping (not used, Lightning uses val_loss)
        use_early_stopping: Whether to use early stopping
        storage: Optuna storage URL (e.g., "sqlite:///study.db") for distributed optimization
        study_name: Name of the study (required if storage is provided)
        gpu_id: GPU device ID to use (None = use default)
        device: Device string ("cuda" or "cpu", None = auto-detect)
        
    Returns:
        Tuple of (best_params, study)
    """
    from chemprop import data, featurizers, models, nn
    from lightning import pytorch as pl
    from sklearn.metrics import roc_auc_score
    
    # Set GPU device if specified
    if gpu_id is not None:
        set_gpu_device(gpu_id)
    
    # Set device
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create featurizer (SimpleMoleculeMolGraphFeaturizer for D-MPNN)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    
    def objective(trial):
        # Sample hyperparameters from search space
        params = _sample_params(trial, search_space)
        
        # Extract parameters
        batch_size = params.pop("batch_size", 64)
        learning_rate = params.pop("learning_rate", 1e-3)
        weight_decay = params.pop("weight_decay", 0.0)
        dropout = params.pop("dropout", 0.0)
        depth = params.pop("depth", 3)
        hidden_dim = params.pop("hidden_dim", 300)
        ffn_hidden_dim = params.pop("ffn_hidden_dim", 300)
        ffn_num_layers = params.pop("ffn_num_layers", 2)
        
        # Create datasets
        train_dataset = data.MoleculeDataset(train_datapoints, featurizer)
        val_dataset = data.MoleculeDataset(val_datapoints, featurizer)
        
        # Create data loaders
        train_loader = data.build_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = data.build_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Build MPNN model
        mp = nn.BondMessagePassing(
            d_h=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
        agg = nn.MeanAggregation()
        ffn = nn.BinaryClassificationFFN(
            input_dim=mp.output_dim,
            hidden_dim=ffn_hidden_dim,
            n_layers=ffn_num_layers,
            dropout=dropout,
            n_tasks=1,  # binary classification
        )
        mpnn = models.MPNN(
            message_passing=mp,
            agg=agg,
            predictor=ffn,
            batch_norm=True,
            metrics=None,
        )
        
        # Configure optimizer
        mpnn.configure_optimizers = lambda: torch.optim.Adam(
            mpnn.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Setup early stopping
        patience = early_stopping_patience if early_stopping_patience is not None else config.TRAINING_PARAMS["early_stopping_patience"]
        
        callbacks = []
        if use_early_stopping:
            callbacks.append(
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    mode="min",
                    verbose=False,
                )
            )
        
        # Create Lightning trainer
        trainer = pl.Trainer(
            max_epochs=nb_epoch,
            callbacks=callbacks,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator=device_str,
            devices=1,
        )
        
        # Train model
        trainer.fit(mpnn, train_loader, val_loader)
        
        # Get predictions on validation set
        val_preds = trainer.predict(mpnn, val_loader)
        val_proba = torch.sigmoid(torch.cat(val_preds)).squeeze().cpu().numpy()
        
        # Calculate ROC-AUC score
        score = roc_auc_score(val_labels, val_proba)
        
        # Clean up model to free GPU memory
        del mpnn
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return score
    
    # Create or load Optuna study
    if storage is not None:
        # Use persistent storage for distributed optimization
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            load_if_exists=True,  # Allow multiple workers to share the study
        )
    else:
        # In-memory study for single-process optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Return best parameters and study
    return study.best_params, study
