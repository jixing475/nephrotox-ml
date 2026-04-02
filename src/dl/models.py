"""Model factory for DeepChem GCN."""

from typing import List, Optional

import deepchem as dc


def create_gcn_model(
    n_tasks: int = 1,
    mode: str = "classification",
    graph_conv_layers: Optional[List[int]] = None,
    dense_layer_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs
):
    """Create DeepChem GCNModel.
    
    Reference: .cursor/skills/deepchem/scripts/graph_neural_network.py
    
    Args:
        n_tasks: Number of prediction tasks
        mode: 'classification' or 'regression'
        graph_conv_layers: List of hidden layer sizes for graph convolution layers.
                          If None, defaults to [64, 64]
        dense_layer_size: Size of dense layer after graph convolution
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay (L2 regularization) coefficient. If > 0, uses AdamW optimizer.
        **kwargs: Additional arguments passed to GCNModel (batch_size, dropout, etc.)
        
    Returns:
        DeepChem GCNModel instance
    """
    # Extract parameters from kwargs if provided (these take precedence)
    graph_conv_layers = kwargs.pop("graph_conv_layers", graph_conv_layers)
    dense_layer_size = kwargs.pop("dense_layer_size", dense_layer_size)
    learning_rate = kwargs.pop("learning_rate", learning_rate)
    weight_decay = kwargs.pop("weight_decay", weight_decay)
    
    default_params = {
        "batch_size": 128,
        "dropout": 0.0,
        "number_atom_features": 74,  # DGLlife CanonicalAtomFeaturizer produces 74D features
    }
    
    # Set graph_conv_layers if provided
    if graph_conv_layers is not None:
        default_params["graph_conv_layers"] = graph_conv_layers
    
    # Set dense_layer_size if provided
    default_params["dense_layer_size"] = dense_layer_size
    
    # Update with remaining kwargs (these take precedence)
    default_params.update(kwargs)
    
    # Create optimizer with weight_decay support
    # If weight_decay > 0, use AdamW for decoupled weight decay
    # Otherwise, use standard Adam
    if weight_decay > 0:
        optimizer = dc.models.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
    else:
        optimizer = dc.models.optimizers.Adam(learning_rate=learning_rate)
    
    return dc.models.GCNModel(
        n_tasks=n_tasks,
        mode=mode,
        optimizer=optimizer,
        **default_params
    )

