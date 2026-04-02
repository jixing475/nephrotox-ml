"""Utilities for handling DGL graphs in multimodal model.

This module provides glue code for DGL graph operations, reusing logic from dl/ project.
"""

import torch
import dgl
import numpy as np
from typing import List, Dict
from dgllife.utils import CanonicalAtomFeaturizer, mol_to_bigraph


def featurize_smiles_to_dgl(smiles_list: List[str], model_type: str = "GCN") -> List:
    """Featurize SMILES to DGL graphs.
    
    Reused from dl/src/trainer_dgllife.py with simplification for GCN only.
    
    Args:
        smiles_list: List of SMILES strings
        model_type: Model type (only "GCN" supported for now)
        
    Returns:
        List of DGL graphs (None for invalid SMILES)
    """
    from rdkit import Chem
    
    # For GCN, use CanonicalAtomFeaturizer (74D features)
    node_featurizer = CanonicalAtomFeaturizer()
    
    graphs = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            graphs.append(None)
            continue
        
        # Convert to DGL bigraph
        g = mol_to_bigraph(
            mol,
            node_featurizer=node_featurizer,
            edge_featurizer=None,
        )
        graphs.append(g)
    
    return graphs


def graphdata_to_dgl(graph_data):
    """
    Convert DeepChem GraphData to DGL graph.
    
    Args:
        graph_data: DeepChem GraphData object
        
    Returns:
        DGL graph with node features
    """
    # Extract node features and adjacency
    node_features = torch.FloatTensor(graph_data.node_features)
    
    # Get edges from adjacency matrix
    # GraphData stores adjacency as (edge_index, edge_features)
    edge_index = graph_data.edge_index  # Shape: (2, num_edges)
    
    # Create DGL graph
    src = edge_index[0]
    dst = edge_index[1]
    g = dgl.graph((src, dst), num_nodes=len(node_features))
    
    # Add self-loops for better GCN performance
    g = dgl.add_self_loop(g)
    
    return g, node_features


def batch_graphdata_to_dgl(graph_data_list: List):
    """
    Convert a list of DeepChem GraphData to a batched DGL graph.
    
    Args:
        graph_data_list: List of DeepChem GraphData objects
        
    Returns:
        Tuple of (batched_graph, batched_node_features)
    """
    graphs = []
    node_features_list = []
    
    for graph_data in graph_data_list:
        g, node_feats = graphdata_to_dgl(graph_data)
        graphs.append(g)
        node_features_list.append(node_feats)
    
    # Batch graphs
    batched_graph = dgl.batch(graphs)
    
    # Concatenate node features
    batched_node_features = torch.cat(node_features_list, dim=0)
    
    return batched_graph, batched_node_features


def collate_multimodal_batch(batch):
    """
    Custom collate function for DataLoader to handle multimodal data with DGL graphs.
    
    Args:
        batch: List of tuples (sample_dict, label)
        
    Returns:
        Tuple of (inputs_dict, labels_tensor)
    """
    # Separate samples and labels
    samples = [item[0] for item in batch]
    labels = torch.FloatTensor([item[1] for item in batch])
    
    # Stack RDKit and ChemBERTa features
    rdkit_batch = torch.FloatTensor(np.stack([s['rdkit'] for s in samples]))
    chemberta_batch = torch.FloatTensor(np.stack([s['chemberta'] for s in samples]))
    
    # Batch graph data
    graph_data_list = [s['graph'] for s in samples]
    graph_batch, graph_feats = batch_graphdata_to_dgl(graph_data_list)
    
    inputs = {
        'rdkit': rdkit_batch,
        'chemberta': chemberta_batch,
        'graph_batch': graph_batch,
        'graph_feats': graph_feats,
    }
    
    return inputs, labels
