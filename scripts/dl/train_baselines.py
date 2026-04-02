"""
Baseline Training Script for Comparative Analysis

This script trains single-modality baselines to compare with the multimodal fusion model.
Models:
1. RDKit-only (MLP)
2. ChemBERTa-only (MLP on frozen embeddings)
3. Graph-only (GCN)

Usage:
    uv run python train_baselines.py
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Reuse components
from src.featurizers import ChemBERTaFeaturizer, extract_rdkit_features
from src.graph_utils import featurize_smiles_to_dgl
from train_full_dataset import MultimodalDataset, collate_multimodal

# ==============================================================================
# BASELINE MODELS
# ==============================================================================

class RDKitModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, inputs):
        return self.mlp(inputs['rdkit'])

class ChemBERTaModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs['chemberta'])

class GraphModel(nn.Module):
    def __init__(self, input_dim=74, hidden_dims=[64, 64, 64], dropout=0.2):
        super().__init__()
        from dgllife.model import GCN
        from dgl.nn.pytorch import SumPooling
        
        self.gnn = GCN(
            in_feats=input_dim,
            hidden_feats=hidden_dims,
            activation=[nn.ReLU()] * len(hidden_dims),
            dropout=[dropout] * len(hidden_dims),
        )
        self.pooling = SumPooling()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, inputs):
        graph = inputs['graph_batch']
        feats = inputs['graph_feats']
        node_embs = self.gnn(graph, feats)
        graph_emb = self.pooling(graph, node_embs)
        return self.mlp(graph_emb)

# ==============================================================================
# TRAINING UTILS
# ==============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            labels = batch['labels'].to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits.squeeze())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    # Handle NaNs if any
    all_probs = np.nan_to_num(all_probs, nan=0.5)
    
    metrics = {
        'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
        'accuracy': accuracy_score(all_labels, (all_probs > 0.5).astype(int)),
        'f1': f1_score(all_labels, (all_probs > 0.5).astype(int), zero_division=0)
    }
    return metrics

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 80)
    print("BASELINE MODEL TRAINING (RDKit, ChemBERTa, Graph)")
    print("=" * 80)
    
    # 1. Load Data (Copied from train_full_dataset.py)
    base_dir = Path(__file__).parent
    train_csv = base_dir / "input" / "cleaned_data_train_rdkit_desc.csv"
    test_csv = base_dir / "input" / "cleaned_data_test_rdkit_desc.csv"
    output_dir = base_dir / "output"
    
    print("Loading data...")
    # Train
    train_df = pd.read_csv(train_csv)
    smiles_train = train_df['SMILES'].tolist()
    labels_train = train_df['label'].values
    rdkit_cols = [col for col in train_df.columns if col not in ['ID', 'SMILES', 'label']]
    rdkit_train = extract_rdkit_features(train_df, rdkit_cols)
    chemberta_feat = ChemBERTaFeaturizer()
    chemberta_train = chemberta_feat.featurize(smiles_train)
    graphs_train = featurize_smiles_to_dgl(smiles_train, model_type="GCN")
    
    # Filter
    valid_idxs = [i for i, g in enumerate(graphs_train) if g is not None]
    graphs_train = [graphs_train[i] for i in valid_idxs]
    rdkit_train = rdkit_train[valid_idxs]
    chemberta_train = chemberta_train[valid_idxs]
    labels_train = labels_train[valid_idxs]
    
    # Test
    test_df = pd.read_csv(test_csv)
    smiles_test = test_df['SMILES'].tolist()
    labels_test = test_df['label'].values
    rdkit_test = extract_rdkit_features(test_df, rdkit_cols)
    chemberta_test = chemberta_feat.featurize(smiles_test)
    graphs_test = featurize_smiles_to_dgl(smiles_test, model_type="GCN")
    
    valid_idxs_test = [i for i, g in enumerate(graphs_test) if g is not None]
    graphs_test = [graphs_test[i] for i in valid_idxs_test]
    rdkit_test = rdkit_test[valid_idxs_test]
    chemberta_test = chemberta_test[valid_idxs_test]
    labels_test = labels_test[valid_idxs_test]
    
    # Standardize RDKit
    scaler = StandardScaler()
    rdkit_train = scaler.fit_transform(rdkit_train)
    rdkit_test = scaler.transform(rdkit_test)
    
    # Create Loaders
    train_ds = MultimodalDataset(rdkit_train, chemberta_train, graphs_train, labels_train)
    test_ds = MultimodalDataset(rdkit_test, chemberta_test, graphs_test, labels_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_multimodal)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Data ready. Train: {len(train_ds)}, Test: {len(test_ds)}")
    
    results = {}
    
    # 2. Train Loop for each modality
    models = {
        'RDKit': RDKitModel(input_dim=rdkit_train.shape[1]),
        'ChemBERTa': ChemBERTaModel(input_dim=768),
        'Graph': GraphModel(input_dim=74)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name} Model...")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) # Same LR as fusion
        criterion = nn.BCEWithLogitsLoss()
        
        best_auc = 0.0
        best_metrics = {}
        
        for epoch in range(50): # 50 epochs for baselines
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            metrics = evaluate(model, test_loader, device)
            
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_metrics = metrics
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={loss:.4f}, AUC={metrics['roc_auc']:.4f}")
                
        print(f"✅ {name} Best AUC: {best_auc:.4f}")
        results[name] = best_metrics
        
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"baseline_metrics_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print("="*80)
    print(f"{'Model':<15} {'AUC':<10} {'F1':<10}")
    print("-" * 35)
    for name, m in results.items():
        print(f"{name:<15} {m['roc_auc']:.4f}     {m['f1']:.4f}")
    
if __name__ == "__main__":
    main()
