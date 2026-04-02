"""Metrics calculation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_curve,
    confusion_matrix,
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calculate evaluation metrics; returns values in percentages.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for positive class)
        
    Returns:
        Dictionary of metrics in percentages
    """
    return {
        "AUC": roc_auc_score(y_true, y_proba) * 100,
        "ACC": accuracy_score(y_true, y_pred) * 100,
        "SE": recall_score(y_true, y_pred, pos_label=1) * 100,  # Sensitivity
        "SP": recall_score(y_true, y_pred, pos_label=0) * 100,  # Specificity
        "F1": f1_score(y_true, y_pred) * 100,
        "Kappa": cohen_kappa_score(y_true, y_pred) * 100,
        "MCC": matthews_corrcoef(y_true, y_pred) * 100,
    }


def metrics_to_frame(metrics_list: list[dict], set_name: str, seeds: list[int]) -> pd.DataFrame:
    """Convert list of metrics dictionaries to DataFrame.
    
    Args:
        metrics_list: List of metric dictionaries
        set_name: Name of the dataset (e.g., 'val', 'test')
        seeds: List of seeds corresponding to each metrics dict
        
    Returns:
        DataFrame with seed, set, and metric columns
    """
    rows = []
    for seed, m in zip(seeds, metrics_list):
        row = {"seed": seed, "set": set_name}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)
