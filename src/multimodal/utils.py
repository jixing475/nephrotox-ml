"""Utility functions for multimodal fusion model."""

import json
import pathlib
from typing import Dict, Any


def save_config(config: Dict[str, Any], output_path: pathlib.Path):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Configuration file path
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def standardize_features(features, mean=None, std=None):
    """
    Standardize features using z-score normalization.
    
    Args:
        features: Feature array
        mean: Pre-computed mean (if None, compute from features)
        std: Pre-computed std (if None, compute from features)
        
    Returns:
        Tuple of (standardized_features, mean, std)
    """
    import numpy as np
    
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        # Avoid division by zero
        std[std == 0] = 1.0
    
    standardized = (features - mean) / std
    return standardized, mean, std
