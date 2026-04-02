"""Featurizers for DeepChem models."""

import os
import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import deepchem as dc
from deepchem.feat.graph_data import GraphData
from joblib import Parallel, delayed
from rdkit import Chem
from dgllife.utils import CanonicalAtomFeaturizer, mol_to_bigraph


def _featurize_single(smiles: str, featurizer_instance):
    """Featurize a single SMILES string (for parallel processing)."""
    try:
        return featurizer_instance._featurize_single(smiles)
    except Exception:
        return None


class GraphFeaturizer:
    """DGLlife CanonicalAtomFeaturizer wrapper (74D features) with parallel processing.
    
    Replaces DeepChem's MolGraphConvFeaturizer (30D) with DGLlife's CanonicalAtomFeaturizer
    (74D) to match paper implementation. Keeps caching and parallel processing capabilities.
    
    Reference: .cursor/plans/gcn_74d_featurizer_fix_11b1ed4e.plan.md
    """
    
    def __init__(self, n_jobs: int = -1, cache_dir: Optional[Path] = None):
        """Initialize featurizer with parallel processing support.
        
        Args:
            n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = no parallelism)
            cache_dir: Optional directory to cache featurized data
        """
        # DGLlife CanonicalAtomFeaturizer for 74D atom features
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.n_jobs = n_jobs
        self.cache_dir = cache_dir
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, smiles_list) -> str:
        """Generate cache key from SMILES list."""
        content = "\n".join(sorted(smiles_list))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str):
        """Load featurized data from cache."""
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, features):
        """Save featurized data to cache."""
        if self.cache_dir is None:
            return
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(features, f)
    
    def _featurize_single(self, smiles: str):
        """Featurize a single SMILES string to DeepChem GraphData.
        
        Args:
            smiles: SMILES string
            
        Returns:
            DeepChem GraphData object with 74D atom features
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Convert to DGL graph with CanonicalAtomFeaturizer (74D features)
        g = mol_to_bigraph(mol, node_featurizer=self.atom_featurizer)
        
        # Convert DGL graph to DeepChem GraphData format
        node_features = g.ndata['h'].numpy()  # Shape: (n_atoms, 74)
        edge_index = np.stack(g.edges(), axis=0)  # Shape: (2, n_edges)
        
        return GraphData(
            node_features=node_features,
            edge_index=edge_index
        )
    
    def featurize(self, smiles_list, use_parallel: bool = True):
        """Convert SMILES to DeepChem graph features with parallel processing.
        
        Args:
            smiles_list: List or array of SMILES strings
            use_parallel: Whether to use parallel processing (default: True)
            
        Returns:
            Array of GraphData objects with 74D atom features
        """
        smiles_list = list(smiles_list)
        
        # Try to load from cache
        if self.cache_dir:
            cache_key = self._get_cache_key(smiles_list)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                print(f"Loaded {len(cached)} features from cache")
                return cached
        
        # Parallel featurization
        if use_parallel and self.n_jobs != 1 and len(smiles_list) > 100:
            print(f"Parallel featurizing {len(smiles_list)} molecules with {self.n_jobs} jobs...")
            features = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_featurize_single)(s, self) for s in smiles_list
            )
            features = np.array([f for f in features if f is not None])
        else:
            # Serial featurization (small datasets or n_jobs=1)
            features = [self._featurize_single(s) for s in smiles_list]
            features = np.array([f for f in features if f is not None])
        
        # Save to cache
        if self.cache_dir:
            self._save_to_cache(cache_key, features)
            print(f"Cached {len(features)} features")
        
        return features
