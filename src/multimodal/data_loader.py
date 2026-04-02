"""Data loading and alignment for multimodal fusion model.

This module provides glue code to combine features from different sources:
- CSV files with RDKit descriptors
- ChemBERTa embeddings
- Graph features
"""

import pandas as pd
import numpy as np
import deepchem as dc
from typing import Tuple, List
from .featurizers import ChemBERTaFeaturizer, GraphFeaturizer, extract_rdkit_features
from .config import CHEMBERTA_CONFIG


def load_multimodal_dataset(
    csv_path: str,
    smiles_col: str = "smiles",
    label_col: str = "label",
    rdkit_feature_cols: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and align multimodal features from CSV.
    
    Args:
        csv_path: Path to CSV file with SMILES and RDKit descriptors
        smiles_col: Column name for SMILES strings
        label_col: Column name for labels
        rdkit_feature_cols: List of RDKit descriptor column names (if None, auto-detect)
        
    Returns:
        Tuple of (rdkit_features, chemberta_features, graph_features, labels)
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    smiles_list = df[smiles_col].tolist()
    labels = df[label_col].values
    
    # Auto-detect RDKit columns if not provided
    if rdkit_feature_cols is None:
        # Assume all numeric columns except label are RDKit features
        rdkit_feature_cols = [
            col for col in df.columns 
            if col not in [smiles_col, label_col] and df[col].dtype in ['float64', 'int64']
        ]
    
    # Extract RDKit features
    rdkit_features = extract_rdkit_features(df, rdkit_feature_cols)
    
    # Extract ChemBERTa features
    print("Extracting ChemBERTa embeddings...")
    chemberta_featurizer = ChemBERTaFeaturizer(
        model_name=CHEMBERTA_CONFIG["model_name"],
        max_length=CHEMBERTA_CONFIG["max_length"]
    )
    chemberta_features = chemberta_featurizer.featurize(smiles_list)
    
    # Extract Graph features
    print("Extracting graph features...")
    graph_featurizer = GraphFeaturizer()
    graph_features = graph_featurizer.featurize(smiles_list)
    
    return rdkit_features, chemberta_features, graph_features, labels


def create_deepchem_dataset(
    rdkit_features: np.ndarray,
    chemberta_features: np.ndarray,
    graph_features: List,
    labels: np.ndarray,
) -> dc.data.NumpyDataset:
    """
    Create DeepChem NumpyDataset from multimodal features.
    
    The dataset X will be a structured array containing all three feature types.
    
    Args:
        rdkit_features: RDKit descriptor array (n_samples, n_rdkit_features)
        chemberta_features: ChemBERTa embedding array (n_samples, 768)
        graph_features: List of GraphData objects
        labels: Label array (n_samples,)
        
    Returns:
        DeepChem NumpyDataset
    """
    # Package features as a list of tuples for each sample
    # Each sample: (rdkit_vec, chemberta_vec, graph_data)
    X = []
    for i in range(len(labels)):
        X.append({
            'rdkit': rdkit_features[i],
            'chemberta': chemberta_features[i],
            'graph': graph_features[i]
        })
    
    # Convert to object array for DeepChem
    X = np.array(X, dtype=object)
    
    dataset = dc.data.NumpyDataset(X=X, y=labels)
    return dataset


def split_dataset(
    dataset: dc.data.NumpyDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[dc.data.NumpyDataset, dc.data.NumpyDataset, dc.data.NumpyDataset]:
    """
    Split dataset into train/val/test.
    
    Args:
        dataset: DeepChem dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    splitter = dc.splits.RandomSplitter()
    train_dataset, val_dataset, test_dataset = splitter.train_valid_test_split(
        dataset,
        frac_train=train_ratio,
        frac_valid=val_ratio,
        frac_test=test_ratio,
        seed=seed
    )
    
    return train_dataset, val_dataset, test_dataset
