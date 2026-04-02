"""Feature extraction utilities for multimodal fusion.

This module provides glue code to extract features from different modalities:
- ChemBERTa embeddings from SMILES
- RDKit descriptors (reused from ml/)
- Graph features (reused from dl/)
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List
import deepchem as dc


class ChemBERTaFeaturizer:
    """Extract ChemBERTa embeddings from SMILES strings."""
    
    def __init__(self, model_name: str = "seyonec/PubChem10M_SMILES_BPE_450k", max_length: int = 128):
        """
        Initialize ChemBERTa featurizer.
        
        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum SMILES sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        
    def _load_model(self):
        """Lazy load model to avoid loading during import."""
        if self.model is None:
            print(f"Loading ChemBERTa model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()  # Set to evaluation mode
            
    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """
        Extract [CLS] token embeddings from SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            numpy array of shape (n_molecules, 768)
        """
        self._load_model()
        
        # Tokenize SMILES
        inputs = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Take [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return embeddings


class GraphFeaturizer:
    """DGLlife CanonicalAtomFeaturizer wrapper (74D features).
    
    Uses DGLlife's mol_to_bigraph with CanonicalAtomFeaturizer (74D) to match
    the dl/ implementation and enable proper DGL GCN integration.
    """
    
    def __init__(self):
        """Initialize graph featurizer using DGLlife CanonicalAtomFeaturizer."""
        from dgllife.utils import CanonicalAtomFeaturizer
        from rdkit import Chem
        # Use CanonicalAtomFeaturizer (74D) for DGL compatibility
        self.atom_featurizer = CanonicalAtomFeaturizer()
        
    def _featurize_single(self, smiles: str):
        """Featurize a single SMILES string to DeepChem GraphData.
        
        Args:
            smiles: SMILES string
            
        Returns:
            DeepChem GraphData object with 74D atom features
        """
        from dgllife.utils import mol_to_bigraph
        from rdkit import Chem
        import deepchem as dc
        import numpy as np
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Convert to DGL graph with CanonicalAtomFeaturizer (74D features)
        g = mol_to_bigraph(mol, node_featurizer=self.atom_featurizer)
        
        # Convert DGL graph to DeepChem GraphData format
        node_features = g.ndata['h'].numpy()  # Shape: (n_atoms, 74)
        edge_index = np.stack(g.edges(), axis=0)  # Shape: (2, n_edges)
        
        return dc.feat.graph_data.GraphData(
            node_features=node_features,
            edge_index=edge_index
        )
        
    def featurize(self, smiles_list: List[str]):
        """
        Convert SMILES to DGL graph features.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of GraphData objects with 74D atom features
        """
        features = [self._featurize_single(s) for s in smiles_list]
        # Filter out None values (failed featurizations)
        features = [f for f in features if f is not None]
        return features


def extract_rdkit_features(df, feature_columns):
    """
    Extract RDKit descriptor features from DataFrame with data cleaning.
    
    This is glue code that reuses the preprocessing logic from ml/.
    Adds NaN/Inf handling to prevent numerical instability.
    
    Args:
        df: pandas DataFrame with RDKit descriptor columns
        feature_columns: List of column names to extract
        
    Returns:
        numpy array of shape (n_molecules, n_features)
    """
    import numpy as np
    
    features = df[feature_columns].values
    
    # Clean NaN and Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip extreme outliers (beyond 99.9th percentile) to prevent numerical issues
    for i in range(features.shape[1]):
        col = features[:, i]
        if col.std() > 0:  # Only clip if there's variance
            p999 = np.percentile(np.abs(col), 99.9)
            features[:, i] = np.clip(col, -p999, p999)
    
    return features
