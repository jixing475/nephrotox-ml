"""DGLlife trainer with PyTorch training loop for GCN, GAT, Weave, and AttentiveFP."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import dgl
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    WeaveAtomFeaturizer,
    WeaveEdgeFeaturizer,
    AttentiveFPAtomFeaturizer,
    AttentiveFPBondFeaturizer,
    mol_to_bigraph,
    smiles_to_bigraph,
    smiles_to_complete_graph,
)

from . import config
from .data_loader import load_data
from .metrics import calculate_metrics
from .models_dgllife import (
    create_dgllife_gcn_model,
    create_dgllife_gat_model,
    create_dgllife_weave_model,
    create_dgllife_attentivefp_model,
)


class DGLGraphDataset:
    """Dataset wrapper for DGL graphs."""
    
    def __init__(self, graphs: list, labels: np.ndarray):
        """Initialize dataset.
        
        Args:
            graphs: List of DGL graphs
            labels: Array of labels (0/1 for binary classification)
        """
        self.graphs = graphs
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def collate_graphs(batch):
    """Collate function for DGL graphs.
    
    Args:
        batch: List of (graph, label) tuples
        
    Returns:
        Batched graph and labels tensor
    """
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)
    return batched_graph, labels


def get_featurizers(model_type: str) -> Tuple:
    """Return (node_featurizer, edge_featurizer) for model type.
    
    Args:
        model_type: Model type ("GCN", "GAT", "Weave", "AttentiveFP")
        
    Returns:
        Tuple of (node_featurizer, edge_featurizer)
    """
    if model_type in ["GCN", "GAT"]:
        return CanonicalAtomFeaturizer(), None
    elif model_type == "Weave":
        return WeaveAtomFeaturizer(), WeaveEdgeFeaturizer()
    elif model_type == "AttentiveFP":
        return AttentiveFPAtomFeaturizer(), AttentiveFPBondFeaturizer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def featurize_smiles_to_dgl(
    smiles_list: list[str],
    model_type: str = "GCN",
    use_bond_features: bool = False,
):
    """Featurize SMILES to DGL graphs.
    
    Args:
        smiles_list: List of SMILES strings
        model_type: Model type ("GCN", "GAT", "Weave", "AttentiveFP")
        use_bond_features: Whether to use bond features (deprecated, use model_type instead)
        
    Returns:
        List of DGL graphs (None for invalid SMILES)
    """
    node_featurizer, edge_featurizer = get_featurizers(model_type)
    
    from rdkit import Chem
    
    graphs = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            graphs.append(None)
            continue
        
        # Use smiles_to_complete_graph for Weave (requires all atom pairs)
        if model_type == "Weave":
            g = smiles_to_complete_graph(
                smiles,
                add_self_loop=True,  # WeaveEdgeFeaturizer includes self-loops
                node_featurizer=node_featurizer,
                edge_featurizer=edge_featurizer,
            )
        # Use smiles_to_bigraph for models with bond-based edge features
        elif edge_featurizer is not None:
            g = smiles_to_bigraph(
                smiles,
                node_featurizer=node_featurizer,
                edge_featurizer=edge_featurizer,
            )
        else:
            g = mol_to_bigraph(
                mol,
                node_featurizer=node_featurizer,
                edge_featurizer=None,
            )
        graphs.append(g)
    
    return graphs


def train_dgllife_gcn_single_seed(
    seed: int,
    df: pd.DataFrame,
    cfg: Dict,
    external_df: pd.DataFrame,
    tuned_params: Dict | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str | None = None,
    use_early_stopping: bool = True,
    device: str | None = None,
):
    """Train DGLlife GCN model for a single seed (backward compatibility wrapper).
    
    Args:
        seed: Random seed
        df: Training dataframe
        cfg: Model configuration
        external_df: External test dataframe
        tuned_params: Optional hyperparameters
        early_stopping_patience: Patience for early stopping
        early_stopping_metric: Metric for early stopping
        use_early_stopping: Whether to use early stopping
        device: Device to use (default: cuda if available, else cpu)
        
    Returns:
        Dictionary with metrics, predictions, and other results (same format as run_single_seed)
    """
    return train_dgllife_single_seed(
        seed, df, cfg, external_df, tuned_params,
        early_stopping_patience, early_stopping_metric, use_early_stopping, device
    )


def train_dgllife_single_seed(
    seed: int,
    df: pd.DataFrame,
    cfg: Dict,
    external_df: pd.DataFrame,
    tuned_params: Dict | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str | None = None,
    use_early_stopping: bool = True,
    device: str | None = None,
    use_class_weights: bool = False,
    optimize_threshold: bool = False,
):
    """Train DGLlife model (GCN, GAT, Weave, or AttentiveFP) for a single seed.
    
    Args:
        seed: Random seed
        df: Training dataframe
        cfg: Model configuration (must contain "algorithm" key)
        external_df: External test dataframe
        tuned_params: Optional hyperparameters
        early_stopping_patience: Patience for early stopping
        early_stopping_metric: Metric for early stopping
        use_early_stopping: Whether to use early stopping
        device: Device to use (default: cuda if available, else cpu)
        use_class_weights: Whether to use class-weighted CrossEntropyLoss for imbalanced data
        optimize_threshold: Whether to optimize classification threshold using Youden's J
        
    Returns:
        Dictionary with metrics, predictions, and other results (same format as run_single_seed)
    """
    # Get model type from config
    model_type = cfg.get("algorithm", "GCN")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Split: 80/10/10
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, stratify=df["label"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=1.0 / 9.0,
        stratify=train_val_df["label"],
        random_state=seed,
    )
    
    # Featurize SMILES to DGL graphs
    print(f"Featurizing {len(train_df)} training molecules...")
    train_graphs = featurize_smiles_to_dgl(train_df["SMILES"].values.tolist(), model_type=model_type)
    # Filter out None graphs and corresponding labels
    train_valid_indices = [i for i, g in enumerate(train_graphs) if g is not None]
    train_graphs = [train_graphs[i] for i in train_valid_indices]
    train_labels = train_df.iloc[train_valid_indices]["label"].values
    
    print(f"Featurizing {len(val_df)} validation molecules...")
    val_graphs = featurize_smiles_to_dgl(val_df["SMILES"].values.tolist(), model_type=model_type)
    val_valid_indices = [i for i, g in enumerate(val_graphs) if g is not None]
    val_graphs = [val_graphs[i] for i in val_valid_indices]
    val_labels = val_df.iloc[val_valid_indices]["label"].values
    
    print(f"Featurizing {len(test_df)} test molecules...")
    test_graphs = featurize_smiles_to_dgl(test_df["SMILES"].values.tolist(), model_type=model_type)
    test_valid_indices = [i for i, g in enumerate(test_graphs) if g is not None]
    test_graphs = [test_graphs[i] for i in test_valid_indices]
    test_labels = test_df.iloc[test_valid_indices]["label"].values
    
    print(f"Featurizing {len(external_df)} external molecules...")
    ext_graphs = featurize_smiles_to_dgl(external_df["SMILES"].values.tolist(), model_type=model_type)
    ext_valid_indices = [i for i, g in enumerate(ext_graphs) if g is not None]
    ext_graphs = [ext_graphs[i] for i in ext_valid_indices]
    ext_labels = external_df.iloc[ext_valid_indices]["label"].values
    ext_ids = external_df.iloc[ext_valid_indices]["ID"].values
    
    # Determine if model uses edge features
    uses_edge_features = model_type in ["Weave", "AttentiveFP"]
    
    # Create datasets
    train_dataset = DGLGraphDataset(train_graphs, train_labels)
    val_dataset = DGLGraphDataset(val_graphs, val_labels)
    test_dataset = DGLGraphDataset(test_graphs, test_labels)
    ext_dataset = DGLGraphDataset(ext_graphs, ext_labels)
    
    # Get hyperparameters
    params = tuned_params.copy() if tuned_params else {}
    batch_size = params.pop("batch_size", 128)
    learning_rate = params.pop("learning_rate", 0.001)
    weight_decay = params.pop("weight_decay", 0.0)
    dropout = params.pop("dropout", 0.0)
    
    # Create model based on type
    if model_type == "GCN":
        # Support both naming conventions: "graph_conv_layers" (from tuning) and "hidden_feats" (direct)
        hidden_feats = params.pop("graph_conv_layers", params.pop("hidden_feats", [64, 64, 64]))
        # Support both naming conventions: "dense_layer_size" (from tuning) and "classifier_hidden_feats" (direct)
        classifier_hidden_feats = params.pop("dense_layer_size", params.pop("classifier_hidden_feats", 128))
        model = create_dgllife_gcn_model(
            in_feats=74,  # CanonicalAtomFeaturizer produces 74D features
            hidden_feats=hidden_feats,
            classifier_hidden_feats=classifier_hidden_feats,
            dropout=dropout,
            pooling="sum",
        ).to(device)
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
        ).to(device)
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
        ).to(device)
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
        ).to(device)
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )
    ext_loader = DataLoader(
        ext_dataset,
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
    
    # Calculate class weights if enabled
    if use_class_weights:
        class_counts = np.bincount(train_labels)
        # Inverse frequency weighting
        class_weights = 1.0 / class_counts.astype(float)
        class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop with early stopping
    patience = early_stopping_patience if early_stopping_patience is not None else config.TRAINING_PARAMS["early_stopping_patience"]
    metric = early_stopping_metric if early_stopping_metric is not None else config.TRAINING_PARAMS["early_stopping_metric"]
    eval_interval = config.TRAINING_PARAMS["eval_interval"]
    nb_epoch = config.TRAINING_PARAMS["nb_epoch"]
    
    best_val_score = None
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    maximize = metric == "roc_auc"
    
    for epoch in range(nb_epoch):
        # Training phase
        model.train()
        train_loss = 0.0
        for bg, labels in train_loader:
            bg = bg.to(device)
            labels = labels.to(device)
            
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
            
            train_loss += loss.item()
        
        # Validation phase (at eval_interval or last epoch)
        if (epoch + 1) % eval_interval == 0 or epoch == nb_epoch - 1:
            model.eval()
            val_proba_list = []
            val_labels_list = []
            
            with torch.no_grad():
                for bg, labels in val_loader:
                    bg = bg.to(device)
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
                from sklearn.metrics import roc_auc_score
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
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience and use_early_stopping:
                    print(f"Early stopping at epoch {epoch + 1} (best epoch: {best_epoch + 1})")
                    break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluation on all sets
    model.eval()
    
    def evaluate_loader(loader):
        """Evaluate model on a data loader, returns only probabilities."""
        proba_list = []
        with torch.no_grad():
            for bg, _ in loader:
                bg = bg.to(device)
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
                proba_list.append(proba.cpu().numpy())
        
        proba_array = np.concatenate(proba_list)[:, 1]  # Positive class probability
        return proba_array
    
    # Get predictions (probabilities only)
    val_proba = evaluate_loader(val_loader)
    test_proba = evaluate_loader(test_loader)
    ext_proba = evaluate_loader(ext_loader)
    
    # Determine optimal threshold if enabled
    if optimize_threshold:
        # Use Youden's J statistic to find optimal threshold on validation set
        fpr_val, tpr_val, thresholds_val = roc_curve(val_labels, val_proba)
        j_scores = tpr_val - fpr_val
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds_val[best_idx]
    else:
        optimal_threshold = 0.5
    
    # Apply threshold to get predictions
    val_pred = (val_proba >= optimal_threshold).astype(int)
    test_pred = (test_proba >= optimal_threshold).astype(int)
    ext_pred = (ext_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    val_metrics = calculate_metrics(val_labels, val_pred, val_proba)
    test_metrics = calculate_metrics(test_labels, test_pred, test_proba)
    ext_metrics = calculate_metrics(ext_labels, ext_pred, ext_proba)
    
    # Add threshold info to metrics
    val_metrics["threshold"] = optimal_threshold
    test_metrics["threshold"] = optimal_threshold
    ext_metrics["threshold"] = optimal_threshold
    
    # Predictions dataframe for external test
    preds_df = pd.DataFrame(
        {
            "seed": seed,
            "ID": ext_ids,
            "y_true": ext_labels,
            "y_pred": ext_pred,
            "y_proba": ext_proba,
        }
    )
    
    # ROC data (external test)
    fpr, tpr, thresholds = roc_curve(ext_labels, ext_proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    
    # Confusion matrix (external test)
    cm = confusion_matrix(ext_labels, ext_pred)
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

