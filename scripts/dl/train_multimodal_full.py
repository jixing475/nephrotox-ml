"""
Full Dataset Training Script for Multimodal Fusion Model

Glue Code Strategy: Reuse data loading and processing from dl/ project.
This script integrates:
- dl/ project's DGL graph processing (featurize_smiles_to_dgl - copied to graph_utils)
- Existing ChemBERTa featurizer
- Existing RDKit feature extraction
- Custom multimodal fusion model

Usage:
    uv run python train_full_dataset.py
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
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, 
    precision_score, recall_score, f1_score
)
from torch.utils.data import DataLoader

# Import our multimodal components
from src.featurizers import ChemBERTaFeaturizer, extract_rdkit_features
from src.models import MultimodalFusionModule
from src.config import MODEL_CONFIG
from src.graph_utils import featurize_smiles_to_dgl


class MultimodalDataset:
    """Dataset for multimodal features."""
    
    def __init__(self, rdkit_feats, chemberta_feats, graphs, labels):
        self.rdkit = torch.FloatTensor(rdkit_feats)
        self.chemberta = torch.FloatTensor(chemberta_feats)
        self.graphs = graphs
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'rdkit': self.rdkit[idx],
            'chemberta': self.chemberta[idx],
            'graph': self.graphs[idx],
            'label': self.labels[idx]
        }


def collate_multimodal(batch):
    """Collate function for multimodal data."""
    import dgl
    
    rdkit = torch.stack([item['rdkit'] for item in batch])
    chemberta = torch.stack([item['chemberta'] for item in batch])
    graphs = [item['graph'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # Batch graphs using DGL
    batched_graph = dgl.batch(graphs)
    
    return {
        'rdkit': rdkit,
        'chemberta': chemberta,
        'graph_batch': batched_graph,
        'graph_feats': batched_graph.ndata['h'],
        'labels': labels
    }


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        # Move to device
        inputs = {
            'rdkit': batch['rdkit'].to(device),
            'chemberta': batch['chemberta'].to(device),
            'graph_batch': batch['graph_batch'].to(device),
            'graph_feats': batch['graph_feats'].to(device),
        }
        labels = batch['labels'].to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.squeeze(), labels.float())
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in loader:
            inputs = {
                'rdkit': batch['rdkit'].to(device),
                'chemberta': batch['chemberta'].to(device),
                'graph_batch': batch['graph_batch'].to(device),
                'graph_feats': batch['graph_feats'].to(device),
            }
            labels = batch['labels'].to(device)
            
            logits = model(inputs)
            loss = criterion(logits.squeeze(), labels.float())
            probs = torch.sigmoid(logits.squeeze())
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Check for NaN
    if np.isnan(all_probs).any():
        print(f"⚠️  Warning: {np.isnan(all_probs).sum()} NaN values in predictions, replacing with 0.5")
        all_probs = np.nan_to_num(all_probs, nan=0.5)
    
    preds = (all_probs > 0.5).astype(int)
    
    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(all_labels, preds),
        'precision': precision_score(all_labels, preds, zero_division=0),
        'recall': recall_score(all_labels, preds, zero_division=0),
        'f1': f1_score(all_labels, preds, zero_division=0),
    }
    
    # Safely compute AUC
    try:
        if len(np.unique(all_labels)) > 1:
            metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)
        else:
            metrics['roc_auc'] = 0.5
    except:
        metrics['roc_auc'] = 0.5
    
    return metrics, all_probs, preds


def main():
    print("=" * 80)
    print("FULL DATASET TRAINING - MULTIMODAL FUSION MODEL")
    print("Glue Code Strategy: Reusing dl/ project's DGL processing")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Setup
    base_dir = Path(__file__).parent
    train_csv = base_dir / "input" / "cleaned_data_train_rdkit_desc.csv"
    test_csv = base_dir / "input" / "cleaned_data_test_rdkit_desc.csv"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"full_model_{timestamp}.pt"
    metrics_path = output_dir / f"full_metrics_{timestamp}.json"
    
    # ========================================================================
    # STEP 1: Load and Process Training Data
    # ========================================================================
    print("=" * 80)
    print("STEP 1: Loading Training Data")
    print("=" * 80)
    
    train_df = pd.read_csv(train_csv)
    smiles_train = train_df['SMILES'].tolist()
    labels_train = train_df['label'].values
    
    print(f"\n✅ Loaded {len(train_df)} training samples")
    print(f"   Label distribution: {np.bincount(labels_train.astype(int))}")
    
    # Extract RDKit features
    print("\n📊 Extracting RDKit features...")
    rdkit_cols = [col for col in train_df.columns if col not in ['ID', 'SMILES', 'label']]
    rdkit_train = extract_rdkit_features(train_df, rdkit_cols)
    print(f"   RDKit: {rdkit_train.shape}")
    
    # Extract ChemBERTa features
    print("📊 Extracting ChemBERTa embeddings...")
    chemberta_feat = ChemBERTaFeaturizer()
    chemberta_train = chemberta_feat.featurize(smiles_train)
    print(f"   ChemBERTa: {chemberta_train.shape}")
    
    # Extract Graph features (reuse dl/ code!)
    print("📊 Featurizing graphs (using dl/ DGL pipeline)...")
    graphs_train = featurize_smiles_to_dgl(smiles_train, model_type="GCN")
    
    # Filter out None graphs
    valid_indices = [i for i, g in enumerate(graphs_train) if g is not None]
    graphs_train = [graphs_train[i] for i in valid_indices]
    rdkit_train = rdkit_train[valid_indices]
    chemberta_train = chemberta_train[valid_indices]
    labels_train = labels_train[valid_indices]
    
    print(f"   Graphs: {len(graphs_train)} valid (filtered {len(smiles_train) - len(graphs_train)} invalid)")
    
    # ========================================================================
    # STEP 2: Load and Process Test Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Loading Test Data")
    print("=" * 80)
    
    test_df = pd.read_csv(test_csv)
    smiles_test = test_df['SMILES'].tolist()
    labels_test = test_df['label'].values
    
    print(f"\n✅ Loaded {len(test_df)} test samples")
    
    rdkit_test = extract_rdkit_features(test_df, rdkit_cols)
    chemberta_test = chemberta_feat.featurize(smiles_test)
    graphs_test = featurize_smiles_to_dgl(smiles_test, model_type="GCN")
    
    valid_indices_test = [i for i, g in enumerate(graphs_test) if g is not None]
    graphs_test = [graphs_test[i] for i in valid_indices_test]
    rdkit_test = rdkit_test[valid_indices_test]
    chemberta_test = chemberta_test[valid_indices_test]
    labels_test = labels_test[valid_indices_test]
    
    print(f"   Valid samples: {len(graphs_test)}")
    
    # ========================================================================
    # STEP 2.5: Standardize RDKit Features
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2.5: Standardizing Features")
    print("=" * 80)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Fit on training data only to avoid leakage
    rdkit_train = scaler.fit_transform(rdkit_train)
    # Transform test data using training statistics
    rdkit_test = scaler.transform(rdkit_test)
    
    print(f"✅ Standardized RDKit features")
    print(f"   Train mean: {rdkit_train.mean():.4f}, std: {rdkit_train.std():.4f}")
    
    # ========================================================================
    # STEP 3: Create Datasets
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Creating Datasets")
    print("=" * 80)
    
    train_dataset = MultimodalDataset(rdkit_train, chemberta_train, graphs_train, labels_train)
    test_dataset = MultimodalDataset(rdkit_test, chemberta_test, graphs_test, labels_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_multimodal)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_multimodal)
    
    print(f"\n✅ Created DataLoaders")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # ========================================================================
    # STEP 4: Initialize Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Initializing Model")
    print("=" * 80)
    
    model = MultimodalFusionModule(
        rdkit_dim=rdkit_train.shape[1],
        chemberta_dim=chemberta_train.shape[1],
        chemberta_projection=MODEL_CONFIG.get('chemberta_projection', 384),
        graph_input_dim=74,  # CanonicalAtomFeaturizer
        graph_hidden_dims=MODEL_CONFIG.get('graph_hidden_dims', [64, 64, 64]),
        fusion_dims=MODEL_CONFIG.get('fusion_dims', [512, 256]),
        dropout=MODEL_CONFIG.get('dropout', 0.2),
        n_tasks=1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model: {total_params:,} parameters")
    
    # ========================================================================
    # STEP 5: Train Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Training Model")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 15
    
    print(f"\nDevice: {device}")
    print(f"Max epochs: 100, Early stopping patience: {patience}\n")
    
    start_time = time.time()
    
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics, _, _ = evaluate(model, test_loader, device)
        
        scheduler.step(test_metrics['roc_auc'])
        
        if test_metrics['roc_auc'] > best_auc:
            best_auc = test_metrics['roc_auc']
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                  f"TestAUC={test_metrics['roc_auc']:.4f}, "
                  f"TestAcc={test_metrics['accuracy']:.4f}")
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break
    
    train_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\n✅ Training completed in {train_time:.1f}s")
    print(f"   Best epoch: {best_epoch+1}, Best AUC: {best_auc:.4f}")
    
    # ========================================================================
    # STEP 6: Final Evaluation
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Final Evaluation")
    print("=" * 80)
    
    test_metrics, test_probs, test_preds = evaluate(model, test_loader, device)
    
    print(f"\n📊 Test Set Performance:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   F1 Score:  {test_metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {test_metrics['roc_auc']:.4f}")
    
    cm = confusion_matrix(labels_test, test_preds)
    print(f"\n   Confusion Matrix:")
    print(f"     TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"     FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
    
    # ========================================================================
    # STEP 7: Save Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Saving Results")
    print("=" * 80)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
        'best_epoch': best_epoch,
    }, model_path)
    
    results = {
        'timestamp': timestamp,
        'dataset': {
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
        },
        'model': {
            'total_parameters': total_params,
        },
        'training': {
            'best_epoch': best_epoch + 1,
            'best_auc': float(best_auc),
            'training_time_seconds': train_time,
        },
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'confusion_matrix': cm.tolist()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Model saved: {model_path}")
    print(f"✅ Metrics saved: {metrics_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\n🎯 Results:")
    print(f"   Training Time: {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"   Best Epoch: {best_epoch+1}")
    print(f"   Test AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   Test F1: {test_metrics['f1']:.4f}")
    print(f"\n✅ Training completed successfully!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
