"""Data loading utilities for DeepChem."""

from __future__ import annotations

import deepchem as dc
import numpy as np
import pandas as pd
from pathlib import Path


def load_data(csv_path: Path | str) -> pd.DataFrame:
    """Load CSV with ID, SMILES, label columns."""
    df = pd.read_csv(csv_path)
    assert "ID" in df.columns, "CSV must contain 'ID' column"
    assert "SMILES" in df.columns, "CSV must contain 'SMILES' column"
    assert "label" in df.columns, "CSV must contain 'label' column"
    return df


def create_deepchem_dataset(smiles: np.ndarray | list, labels: np.ndarray | list, featurizer) -> dc.data.NumpyDataset:
    """Create DeepChem NumpyDataset from SMILES and labels.
    
    Args:
        smiles: Array or list of SMILES strings
        labels: Array or list of labels (must be integers 0/1 for classification)
        featurizer: DeepChem featurizer instance
        
    Returns:
        DeepChem NumpyDataset
    """
    X = featurizer.featurize(smiles)
    # Ensure labels are integers for classification (not floats)
    y = np.array(labels, dtype=np.int64).reshape(-1, 1)
    return dc.data.NumpyDataset(X=X, y=y)
