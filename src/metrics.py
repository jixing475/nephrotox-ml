from __future__ import annotations

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
import pandas as pd


def calculate_metrics(y_true, y_pred, y_proba) -> dict:
    """Calculate evaluation metrics; returns values in percentages."""
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
    rows = []
    for seed, m in zip(seeds, metrics_list):
        row = {"seed": seed, "set": set_name}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)
