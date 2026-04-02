"""Chemprop D-MPNN trainer with PyTorch Lightning.

This module provides glue code to integrate chemprop v2 into the existing
dl framework. It wraps chemprop's API without reimplementing its functionality.
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Chemprop v2 imports
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl

from . import config
from .metrics import calculate_metrics


class DescriptorPreprocessor:
    """Preprocessor for molecular descriptors in fusion mode.
    
    Applies feature engineering steps similar to ML module but adapted for DL:
    1. Inf/NaN handling and median imputation
    2. Optional variance filtering (removes zero-variance features)
    3. Optional correlation filtering (removes highly correlated features)
    4. Standardization (applied separately via StandardScaler)
    
    Note: Unlike ML module, we skip numerical clipping as DL models with
    batch normalization are more robust to extreme values.
    """
    
    def __init__(self, variance_threshold: float = 0.0, corr_threshold: float = 0.95):
        """Initialize preprocessor.
        
        Args:
            variance_threshold: Threshold for variance filtering (0.0 = remove only zero-variance)
            corr_threshold: Threshold for correlation filtering (0.95 = remove |r| > 0.95)
        """
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        
        # Fitted parameters
        self.medians: pd.Series | None = None
        self.selected_features: list[str] | None = None
        self.feature_cols: list[str] | None = None
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray | None:
        """Fit preprocessor on training data and transform.
        
        Args:
            df: Training DataFrame with descriptor columns
            
        Returns:
            Preprocessed descriptor array, or None if no descriptors found
        """
        exclude_cols = {"ID", "SMILES", "label"}
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        if not self.feature_cols:
            return None
        
        X = df[self.feature_cols].copy()
        n_initial = X.shape[1]
        
        # Step 1: Handle inf values and impute missing
        X = X.replace([np.inf, -np.inf], np.nan)
        self.medians = X.median()
        X = X.fillna(self.medians).fillna(0)
        
        # Step 2: Variance filtering (optional, remove zero-variance features)
        if self.variance_threshold >= 0:
            variances = X.var()
            high_var_cols = variances[variances > self.variance_threshold].index.tolist()
            n_var_removed = len(X.columns) - len(high_var_cols)
            X = X[high_var_cols]
            if n_var_removed > 0:
                print(f"  Variance filtering: removed {n_var_removed} low-variance features")
        
        # Step 3: Correlation filtering (optional, remove highly correlated features)
        if self.corr_threshold < 1.0 and X.shape[1] > 1:
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            cols_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > self.corr_threshold)]
            X = X.drop(columns=cols_to_drop)
            if cols_to_drop:
                print(f"  Correlation filtering: removed {len(cols_to_drop)} highly correlated features")
        
        self.selected_features = X.columns.tolist()
        
        if X.shape[1] < n_initial:
            print(f"  Feature selection: {n_initial} -> {X.shape[1]} features")
        
        return X.values.astype(np.float32)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray | None:
        """Transform validation/test/external data using fitted parameters.
        
        Args:
            df: DataFrame with descriptor columns
            
        Returns:
            Preprocessed descriptor array, or None if no descriptors found
        """
        if self.medians is None or self.selected_features is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform()")
        
        if not self.feature_cols:
            return None
        
        X = df[self.feature_cols].copy()
        
        # Step 1: Handle inf values and impute using training medians
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.medians).fillna(0)
        
        # Step 2 & 3: Apply feature selection from training
        X = X[self.selected_features]
        
        return X.values.astype(np.float32)


def _extract_molecular_descriptors(df: pd.DataFrame, medians: pd.Series | None = None) -> tuple[np.ndarray | None, pd.Series | None]:
    """Legacy function for backward compatibility. Use DescriptorPreprocessor for new code.
    
    Extract and preprocess molecular descriptor columns from DataFrame.
    
    Works with any descriptor type (RDKit, ChemoPy2d, etc.).
    Excludes: ID, SMILES, label columns.
    Returns None if no descriptor columns found.
    
    Preprocessing steps:
    1. Replace inf/-inf with NaN
    2. Median imputation for missing values (computed from training set)
    
    Args:
        df: DataFrame with descriptor columns
        medians: Pre-computed medians from training set (for val/test/external sets)
                 If None, compute medians from this DataFrame (for training set)
    
    Returns:
        Tuple of (descriptor array, medians). Returns (None, None) if no descriptors found.
    """
    exclude_cols = {"ID", "SMILES", "label"}
    desc_cols = [c for c in df.columns if c not in exclude_cols]
    if not desc_cols:
        return None, None
    
    X = df[desc_cols].copy()
    
    # Step 1: Handle inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Step 2: Median imputation for missing values
    if medians is None:
        # Training set: compute medians
        medians = X.median()
    
    # Apply medians, filling any remaining NaN with 0
    X = X.fillna(medians).fillna(0)
    
    return X.values.astype(np.float32), medians


def train_chemprop_single_seed(
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
    """Train Chemprop D-MPNN model for a single seed.
    
    Args:
        seed: Random seed
        df: Training dataframe with SMILES and label columns
        cfg: Model configuration
        external_df: External test dataframe
        tuned_params: Optional hyperparameters
        early_stopping_patience: Patience for early stopping
        early_stopping_metric: Metric for early stopping (not used, Lightning uses val_loss)
        use_early_stopping: Whether to use early stopping
        device: Device to use (default: cuda if available, else cpu)
        use_class_weights: Not implemented for chemprop (ignored)
        optimize_threshold: Whether to optimize classification threshold
        
    Returns:
        Dictionary with metrics, predictions, and other results
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    # Detect fusion mode from config feature_type
    use_fusion = cfg.get("feature_type", "Graph").startswith("Graph+")
    
    # Extract and scale molecular descriptors if fusion mode
    if use_fusion:
        descriptor_type = cfg.get("feature_type", "Graph").split("+")[1] if "+" in cfg.get("feature_type", "Graph") else "descriptors"
        print(f"Extracting {descriptor_type} descriptors for feature fusion...")
        
        # Initialize preprocessor with feature engineering options
        preprocessor_cfg = config.DESCRIPTOR_PREPROCESSING
        variance_thresh = preprocessor_cfg["variance_threshold"] if preprocessor_cfg["enable_variance_filter"] else -1
        corr_thresh = preprocessor_cfg["corr_threshold"] if preprocessor_cfg["enable_corr_filter"] else 1.0
        
        preprocessor = DescriptorPreprocessor(
            variance_threshold=variance_thresh,
            corr_threshold=corr_thresh
        )
        
        # Fit on training data and transform
        extra_train = preprocessor.fit_transform(train_df)
        
        # Transform validation/test/external sets using fitted parameters
        extra_val = preprocessor.transform(val_df)
        extra_test = preprocessor.transform(test_df)
        extra_ext = preprocessor.transform(external_df)
        
        # Scale descriptors using StandardScaler fitted on training set
        scaler = StandardScaler()
        extra_train = scaler.fit_transform(extra_train)
        extra_val = scaler.transform(extra_val)
        extra_test = scaler.transform(extra_test)
        extra_ext = scaler.transform(extra_ext)
        
        print(f"Final descriptor dimension: {extra_train.shape[1]}")
    else:
        extra_train = extra_val = extra_test = extra_ext = None
    
    # Convert DataFrames to chemprop MoleculeDatapoint
    print(f"Converting {len(train_df)} training molecules to MoleculeDatapoint...")
    train_datapoints = _df_to_chemprop_datapoints(train_df, extra_features=extra_train)
    
    print(f"Converting {len(val_df)} validation molecules to MoleculeDatapoint...")
    val_datapoints = _df_to_chemprop_datapoints(val_df, extra_features=extra_val)
    
    print(f"Converting {len(test_df)} test molecules to MoleculeDatapoint...")
    test_datapoints = _df_to_chemprop_datapoints(test_df, extra_features=extra_test)
    
    print(f"Converting {len(external_df)} external molecules to MoleculeDatapoint...")
    ext_datapoints = _df_to_chemprop_datapoints(external_df, extra_features=extra_ext)
    
    # Create featurizer (SimpleMoleculeMolGraphFeaturizer for D-MPNN)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    
    # Create datasets
    train_dataset = data.MoleculeDataset(train_datapoints, featurizer)
    val_dataset = data.MoleculeDataset(val_datapoints, featurizer)
    test_dataset = data.MoleculeDataset(test_datapoints, featurizer)
    ext_dataset = data.MoleculeDataset(ext_datapoints, featurizer)
    
    # Get descriptor dimension for FFN input size configuration
    d_xd = train_dataset.d_xd  # Will be 0 if no extra features
    
    # Get hyperparameters
    params = tuned_params.copy() if tuned_params else {}
    batch_size = params.pop("batch_size", 64)
    learning_rate = params.pop("learning_rate", 1e-3)
    weight_decay = params.pop("weight_decay", 0.0)
    dropout = params.pop("dropout", 0.0)
    depth = params.pop("depth", 3)
    hidden_dim = params.pop("hidden_dim", 300)
    ffn_hidden_dim = params.pop("ffn_hidden_dim", 300)
    ffn_num_layers = params.pop("ffn_num_layers", 2)
    
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
    test_loader = data.build_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    ext_loader = data.build_dataloader(
        ext_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Build MPNN model
    # Message passing
    mp = nn.BondMessagePassing(
        d_h=hidden_dim,
        depth=depth,
        dropout=dropout,
    )
    
    # Aggregation
    agg = nn.MeanAggregation()
    
    # Feed-forward network for binary classification
    # Input dimension includes D-MPNN output + extra descriptors (if any)
    ffn = nn.BinaryClassificationFFN(
        input_dim=mp.output_dim + d_xd,
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_num_layers,
        dropout=dropout,
        n_tasks=1,  # binary classification
    )
    
    # Create MPNN model
    mpnn = models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        metrics=None,  # Use default AUROC
    )
    
    # Configure optimizer (chemprop uses this internally)
    mpnn.configure_optimizers = lambda: torch.optim.Adam(
        mpnn.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Setup early stopping
    patience = early_stopping_patience if early_stopping_patience is not None else config.TRAINING_PARAMS["early_stopping_patience"]
    nb_epoch = config.TRAINING_PARAMS["nb_epoch"]
    
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
        accelerator=device,
        devices=1,
    )
    
    # Train model
    trainer.fit(mpnn, train_loader, val_loader)
    
    # Get predictions (Lightning returns list of batches)
    val_preds = trainer.predict(mpnn, val_loader)
    test_preds = trainer.predict(mpnn, test_loader)
    ext_preds = trainer.predict(mpnn, ext_loader)
    
    # Convert predictions to numpy arrays
    # For binary classification, chemprop returns logits, need to apply sigmoid
    val_proba = torch.sigmoid(torch.cat(val_preds)).squeeze().cpu().numpy()
    test_proba = torch.sigmoid(torch.cat(test_preds)).squeeze().cpu().numpy()
    ext_proba = torch.sigmoid(torch.cat(ext_preds)).squeeze().cpu().numpy()
    
    # Get true labels
    val_labels = val_df["label"].values
    test_labels = test_df["label"].values
    ext_labels = external_df["label"].values
    
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
            "ID": external_df["ID"].values,
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
    
    # Feature importance (not applicable for D-MPNN, return empty DataFrame)
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


def _df_to_chemprop_datapoints(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    label_col: str = "label",
    extra_features: np.ndarray | None = None,
) -> List[data.MoleculeDatapoint]:
    """Convert DataFrame to list of chemprop MoleculeDatapoint.
    
    Args:
        df: DataFrame with SMILES and label columns
        smiles_col: Name of SMILES column
        label_col: Name of label column
        extra_features: Optional extra features for feature fusion (x_d)
        
    Returns:
        List of MoleculeDatapoint objects
    """
    datapoints = []
    for idx, (i, row) in enumerate(df.iterrows()):
        smi = row[smiles_col]
        y = np.array([row[label_col]], dtype=np.float32)  # chemprop expects array
        
        # Create datapoint with optional extra features
        if extra_features is not None:
            dp = data.MoleculeDatapoint.from_smi(smi, y, x_d=extra_features[idx])
        else:
            dp = data.MoleculeDatapoint.from_smi(smi, y)
        
        datapoints.append(dp)
    
    return datapoints
