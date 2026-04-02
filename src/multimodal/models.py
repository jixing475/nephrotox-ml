"""Multimodal fusion model architecture.

This module defines the triple-stream fusion model:
- RDKit Stream: Descriptor features
- ChemBERTa Stream: Language model embeddings
- Graph Stream: Graph neural network features
"""

import torch
import torch.nn as nn
from typing import List, Dict
import deepchem as dc


class MultimodalFusionModule(nn.Module):
    """
    Triple-stream fusion model for molecular property prediction.
    
    Architecture:
        RDKit (134D) ──────────────┐
                                   ├──> Concat (774D) ──> MLP ──> Output
        ChemBERTa (768D) ──> Proj ─┤
                                   │
        Graph ──> GCN ──> Pool ────┘
    """
    
    def __init__(
        self,
        rdkit_dim: int = 134,
        chemberta_dim: int = 768,
        chemberta_projection: int = 384,
        graph_input_dim: int = 30,
        graph_hidden_dims: List[int] = [128, 256],
        fusion_dims: List[int] = [512, 256],
        dropout: float = 0.2,
        n_tasks: int = 1,
    ):
        """
        Initialize multimodal fusion model.
        
        Args:
            rdkit_dim: RDKit descriptor dimension
            chemberta_dim: ChemBERTa embedding dimension
            chemberta_projection: Projection dimension for ChemBERTa
            graph_input_dim: Graph node feature dimension
            graph_hidden_dims: Hidden dimensions for GCN layers
            fusion_dims: Hidden dimensions for fusion MLP
            dropout: Dropout rate
            n_tasks: Number of output tasks
        """
        super().__init__()
        
        # Stream 1: RDKit (direct pass-through with optional normalization)
        self.rdkit_stream = nn.Identity()
        
        # Stream 2: ChemBERTa projection
        self.chemberta_projection = nn.Sequential(
            nn.Linear(chemberta_dim, chemberta_projection),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Stream 3: Graph - DGL GCN with readout pooling
        # Import DGL components for graph processing
        from dgllife.model import GCN
        from dgl.nn.pytorch import SumPooling
        
        # DGL GCN backbone (matches dl/ implementation)
        self.graph_gnn = GCN(
            in_feats=graph_input_dim,  # 74D from CanonicalAtomFeaturizer
            hidden_feats=graph_hidden_dims,
            activation=[nn.ReLU()] * len(graph_hidden_dims),
            dropout=[dropout] * len(graph_hidden_dims),
        )
        
        # Graph readout layer (sum pooling to get graph-level representation)
        self.graph_readout = SumPooling()
        graph_output_dim = graph_hidden_dims[-1]
        
        # Fusion MLP
        combined_dim = rdkit_dim + chemberta_projection + graph_output_dim
        
        fusion_layers = []
        prev_dim = combined_dim
        for hidden_dim in fusion_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        fusion_layers.append(nn.Linear(prev_dim, n_tasks))
        
        self.fusion_mlp = nn.Sequential(*fusion_layers)
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Dictionary with keys 'rdkit', 'chemberta', 'graph_batch', 'graph_feats'
                - rdkit: (batch_size, rdkit_dim)
                - chemberta: (batch_size, chemberta_dim)
                - graph_batch: DGL batched graph
                - graph_feats: Node features tensor (total_nodes, graph_input_dim)
                
        Returns:
            Predictions of shape (batch_size, n_tasks)
        """
        # Process each stream
        f_rdkit = self.rdkit_stream(inputs['rdkit'])
        f_chemberta = self.chemberta_projection(inputs['chemberta'])
        
        # Process graph through GCN and readout pooling
        graph_batch = inputs['graph_batch']
        graph_feats = inputs['graph_feats']
        
        # Graph convolution
        node_embeddings = self.graph_gnn(graph_batch, graph_feats)
        
        # Graph-level pooling (sum pooling across nodes)
        f_graph = self.graph_readout(graph_batch, node_embeddings)
        
        # Concatenate features
        combined = torch.cat([f_rdkit, f_chemberta, f_graph], dim=1)
        
        # Fusion MLP
        output = self.fusion_mlp(combined)
        
        return output


class MultimodalDeepChemModel:
    """
    Wrapper for multimodal fusion model using DeepChem's TorchModel.
    
    This provides glue code to integrate the custom PyTorch model
    with DeepChem's training infrastructure.
    """
    
    def __init__(self, model_config: dict, training_params: dict):
        """
        Initialize DeepChem-wrapped multimodal model.
        
        Args:
            model_config: Model architecture configuration
            training_params: Training parameters
        """
        self.model_config = model_config
        self.training_params = training_params
        
        # Create PyTorch module
        self.pytorch_model = MultimodalFusionModule(
            rdkit_dim=model_config.get('rdkit_dim', 134),
            chemberta_dim=model_config.get('chemberta_dim', 768),
            chemberta_projection=model_config.get('chemberta_projection', 384),
            graph_hidden_dims=model_config.get('graph_hidden_dims', [128, 256]),
            fusion_dims=model_config.get('fusion_dims', [512, 256]),
            dropout=model_config.get('dropout', 0.2),
            n_tasks=1,
        )
        
        # Wrap with DeepChem TorchModel
        # Note: This is a simplified wrapper. Full implementation would need
        # custom data handling for the multimodal inputs
        self.model = None  # Will be initialized during training
        
    def build_model(self):
        """Build DeepChem TorchModel wrapper."""
        # This is a placeholder - actual implementation would require
        # custom data pipeline to handle the multimodal inputs
        raise NotImplementedError(
            "Full DeepChem integration requires custom data pipeline. "
            "Use direct PyTorch training loop for now."
        )
