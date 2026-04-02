"""DGLlife model factories for GCN, GAT, Weave, and AttentiveFP."""

import torch
import torch.nn as nn
from dgllife.model import GCN, GATPredictor, WeavePredictor, AttentiveFPPredictor
from dgl.nn.pytorch import SumPooling, AvgPooling


class DGLlifeGCNClassifier(nn.Module):
    """DGLlife GCN for binary classification.
    
    This model uses DGLlife's native GCN implementation with 74D atom features
    from CanonicalAtomFeaturizer. Matches paper implementation for better
    performance comparison.
    
    Args:
        in_feats: Input feature size (default: 74 for CanonicalAtomFeaturizer)
        hidden_feats: List of hidden layer sizes for GCN layers
        classifier_hidden_feats: Size of hidden layer in classifier
        dropout: Dropout rate
        pooling: Pooling method ("sum" or "avg")
    """
    
    def __init__(
        self,
        in_feats: int = 74,
        hidden_feats: list[int] | None = None,
        classifier_hidden_feats: int = 128,
        dropout: float = 0.0,
        pooling: str = "sum",
    ):
        super().__init__()
        hidden_feats = hidden_feats or [64, 64, 64]
        
        # DGLlife GCN backbone
        self.gnn = GCN(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            activation=[nn.ReLU()] * len(hidden_feats),
            dropout=[dropout] * len(hidden_feats),
        )
        
        # Readout layer
        if pooling == "sum":
            self.readout = SumPooling()
        elif pooling == "avg":
            self.readout = AvgPooling()
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_feats[-1], classifier_hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_feats, 2),  # Binary classification
        )
    
    def forward(self, bg, node_feats):
        """Forward pass.
        
        Args:
            bg: DGL batched graph
            node_feats: Node features tensor of shape (total_nodes, in_feats)
            
        Returns:
            Logits tensor of shape (batch_size, 2)
        """
        # Graph convolution
        node_feats = self.gnn(bg, node_feats)
        
        # Graph-level pooling
        graph_feats = self.readout(bg, node_feats)
        
        # Classification
        logits = self.classifier(graph_feats)
        
        return logits


def create_dgllife_gcn_model(
    in_feats: int = 74,
    hidden_feats: list[int] | None = None,
    classifier_hidden_feats: int = 128,
    dropout: float = 0.0,
    pooling: str = "sum",
    **kwargs
):
    """Create DGLlife GCN model.
    
    Args:
        in_feats: Input feature size (default: 74)
        hidden_feats: List of hidden layer sizes
        classifier_hidden_feats: Classifier hidden layer size
        dropout: Dropout rate
        pooling: Pooling method ("sum" or "avg")
        **kwargs: Additional arguments (ignored for compatibility)
        
    Returns:
        DGLlifeGCNClassifier instance
    """
    return DGLlifeGCNClassifier(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        classifier_hidden_feats=classifier_hidden_feats,
        dropout=dropout,
        pooling=pooling,
    )


def create_dgllife_gat_model(
    in_feats: int = 74,
    hidden_feats: list[int] | None = None,
    num_heads: list[int] | None = None,
    feat_drops: list[float] | None = None,
    attn_drops: list[float] | None = None,
    classifier_hidden_feats: int = 128,
    classifier_dropout: float = 0.0,
    predictor_hidden_feats: int = 128,
    predictor_dropout: float = 0.0,
    **kwargs
):
    """Create DGLlife GAT model using GATPredictor.
    
    Args:
        in_feats: Input feature size (default: 74 for CanonicalAtomFeaturizer)
        hidden_feats: List of hidden layer sizes (default: [64, 64, 64])
        num_heads: List of attention heads per layer (default: [8, 8, 8])
        feat_drops: List of feature dropout rates (default: [0.0] * len(hidden_feats))
        attn_drops: List of attention dropout rates (default: [0.0] * len(hidden_feats))
        classifier_hidden_feats: Classifier hidden layer size
        classifier_dropout: Classifier dropout rate
        predictor_hidden_feats: Predictor hidden layer size
        predictor_dropout: Predictor dropout rate
        **kwargs: Additional arguments (ignored for compatibility)
        
    Returns:
        GATPredictor instance configured for binary classification
    """
    hidden_feats = hidden_feats or [64, 64, 64]
    num_heads = num_heads or [8, 8, 8]
    feat_drops = feat_drops or [0.0] * len(hidden_feats)
    attn_drops = attn_drops or [0.0] * len(hidden_feats)
    
    return GATPredictor(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        num_heads=num_heads,
        feat_drops=feat_drops,
        attn_drops=attn_drops,
        classifier_hidden_feats=classifier_hidden_feats,
        classifier_dropout=classifier_dropout,
        n_tasks=2,  # Binary classification
        predictor_hidden_feats=predictor_hidden_feats,
        predictor_dropout=predictor_dropout,
    )


def create_dgllife_weave_model(
    node_in_feats: int = 27,
    edge_in_feats: int = 12,
    num_gnn_layers: int = 2,
    gnn_hidden_feats: int = 50,
    graph_feats: int = 128,
    gaussian_expand: bool = True,
    **kwargs
):
    """Create DGLlife Weave model using WeavePredictor.
    
    Args:
        node_in_feats: Node input feature size (default: 27 for WeaveAtomFeaturizer)
        edge_in_feats: Edge input feature size (default: 12 for WeaveEdgeFeaturizer)
        num_gnn_layers: Number of GNN layers
        gnn_hidden_feats: Hidden feature size for GNN layers
        graph_feats: Graph-level feature size
        gaussian_expand: Whether to use Gaussian expansion
        **kwargs: Additional arguments (ignored for compatibility)
        
    Returns:
        WeavePredictor instance configured for binary classification
    """
    return WeavePredictor(
        node_in_feats=node_in_feats,
        edge_in_feats=edge_in_feats,
        num_gnn_layers=num_gnn_layers,
        gnn_hidden_feats=gnn_hidden_feats,
        graph_feats=graph_feats,
        gaussian_expand=gaussian_expand,
        n_tasks=2,  # Binary classification
    )


def create_dgllife_attentivefp_model(
    node_feat_size: int = 39,
    edge_feat_size: int = 10,
    num_layers: int = 2,
    num_timesteps: int = 2,
    graph_feat_size: int = 200,
    dropout: float = 0.0,
    **kwargs
):
    """Create DGLlife AttentiveFP model using AttentiveFPPredictor.
    
    Args:
        node_feat_size: Node feature size (default: 39 for AttentiveFPAtomFeaturizer)
        edge_feat_size: Edge feature size (default: 10 for AttentiveFPBondFeaturizer)
        num_layers: Number of GNN layers
        num_timesteps: Number of attention timesteps
        graph_feat_size: Graph-level feature size
        dropout: Dropout rate
        **kwargs: Additional arguments (ignored for compatibility)
        
    Returns:
        AttentiveFPPredictor instance configured for binary classification
    """
    return AttentiveFPPredictor(
        node_feat_size=node_feat_size,
        edge_feat_size=edge_feat_size,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        graph_feat_size=graph_feat_size,
        n_tasks=2,  # Binary classification
        dropout=dropout,
    )

