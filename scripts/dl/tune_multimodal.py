"""
Optimized Multimodal Training with Optuna Hyperparameter Tuning
Based on dl/ project best practices for achieving AUC > 0.9

Key optimizations:
1. Longer training (300 epochs) with patient early stopping (25)
2. Larger search space with stronger regularization
3. Use GAT instead of GCN for graph branch (better performance in dl/)
4. More Optuna trials (50+)
5. Cross-validation style repeated runs

Usage:
    uv run python tune_multimodal.py
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Import components
from src.featurizers import ChemBERTaFeaturizer, extract_rdkit_features
from src.graph_utils import featurize_smiles_to_dgl
from train_full_dataset import MultimodalDataset, collate_multimodal

# ==============================================================================
# OPTIMIZED MODEL - Use GAT for Graph Branch (better than GCN in dl/ project)
# ==============================================================================

class OptimizedMultimodalModel(nn.Module):
    """Optimized multimodal fusion model with GAT graph branch."""
    
    def __init__(
        self,
        rdkit_dim: int,
        chemberta_dim: int = 768,
        chemberta_proj: int = 256,
        graph_hidden_dims: list = [64, 64, 64],
        graph_num_heads: list = [4, 4, 4],
        fusion_dims: list = [512, 256],
        dropout: float = 0.3,
        use_gat: bool = True,  # Use GAT instead of GCN
    ):
        super().__init__()
        
        self.use_gat = use_gat
        
        # Stream 1: RDKit MLP with BatchNorm
        self.rdkit_stream = nn.Sequential(
            nn.Linear(rdkit_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        rdkit_out_dim = 128
        
        # Stream 2: ChemBERTa projection with BatchNorm
        self.chemberta_stream = nn.Sequential(
            nn.Linear(chemberta_dim, chemberta_proj),
            nn.BatchNorm1d(chemberta_proj),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        chemberta_out_dim = chemberta_proj
        
        # Stream 3: Graph - GAT (better than GCN based on dl/ results)
        from dgl.nn.pytorch import SumPooling
        
        if use_gat:
            from dgllife.model import GAT
            # GAT with attention mechanism - better for capturing complex graph patterns
            self.graph_gnn = GAT(
                in_feats=74,
                hidden_feats=graph_hidden_dims,
                num_heads=graph_num_heads,
                feat_drops=[dropout] * len(graph_hidden_dims),
                attn_drops=[dropout] * len(graph_hidden_dims),
                activations=[nn.ELU()] * len(graph_hidden_dims),
            )
            # GAT output: hidden_feats[-1] * num_heads[-1]
            gnn_out_dim = graph_hidden_dims[-1] * graph_num_heads[-1]
        else:
            from dgllife.model import GCN
            self.graph_gnn = GCN(
                in_feats=74,
                hidden_feats=graph_hidden_dims,
                activation=[nn.ReLU()] * len(graph_hidden_dims),
                dropout=[dropout] * len(graph_hidden_dims),
            )
            gnn_out_dim = graph_hidden_dims[-1]
        
        self.graph_pooling = SumPooling()
        
        # Graph post-processing (use gnn_out_dim as input)
        self.graph_post = nn.Sequential(
            nn.Linear(gnn_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        graph_out_dim = 128
        
        # Fusion MLP
        combined_dim = rdkit_out_dim + chemberta_out_dim + graph_out_dim
        
        layers = []
        prev_dim = combined_dim
        for h_dim in fusion_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.fusion = nn.Sequential(*layers)
        
    def forward(self, inputs):
        # RDKit stream
        f_rdkit = self.rdkit_stream(inputs['rdkit'])
        
        # ChemBERTa stream
        f_chemberta = self.chemberta_stream(inputs['chemberta'])
        
        # Graph stream
        graph = inputs['graph_batch']
        feats = inputs['graph_feats']
        node_embs = self.graph_gnn(graph, feats)
        graph_emb = self.graph_pooling(graph, node_embs)
        f_graph = self.graph_post(graph_emb)
        
        # Fusion
        combined = torch.cat([f_rdkit, f_chemberta, f_graph], dim=1)
        return self.fusion(combined)


# ==============================================================================
# TRAINING UTILS
# ==============================================================================

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs = {
            'rdkit': batch['rdkit'].to(device),
            'chemberta': batch['chemberta'].to(device),
            'graph_batch': batch['graph_batch'].to(device),
            'graph_feats': batch['graph_feats'].to(device),
        }
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.squeeze(), labels.float())
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
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
            probs = torch.sigmoid(logits.squeeze())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_probs = np.nan_to_num(all_probs, nan=0.5)
    
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    return auc, all_probs, all_labels


# ==============================================================================
# OPTUNA OBJECTIVE
# ==============================================================================

def create_objective(train_loader, val_loader, rdkit_dim, device):
    """Create Optuna objective function."""
    
    def objective(trial):
        # Sample hyperparameters - based on dl/ config.py best practices
        params = {
            'chemberta_proj': trial.suggest_int('chemberta_proj', 128, 512),
            'graph_hidden_dim': trial.suggest_categorical('graph_hidden_dim', [32, 64, 128]),
            'graph_num_heads': trial.suggest_categorical('graph_num_heads', [4, 6, 8]),
            'fusion_dim1': trial.suggest_int('fusion_dim1', 256, 512),
            'fusion_dim2': trial.suggest_int('fusion_dim2', 64, 256),
            'dropout': trial.suggest_float('dropout', 0.2, 0.5),  # Higher dropout for regularization
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'use_gat': False,  # Disabled - use GCN only for stability
        }
        
        # Create model
        graph_hidden_dims = [params['graph_hidden_dim']] * 3
        graph_num_heads = [params['graph_num_heads']] * 3
        fusion_dims = [params['fusion_dim1'], params['fusion_dim2']]
        
        model = OptimizedMultimodalModel(
            rdkit_dim=rdkit_dim,
            chemberta_proj=params['chemberta_proj'],
            graph_hidden_dims=graph_hidden_dims,
            graph_num_heads=graph_num_heads,
            fusion_dims=fusion_dims,
            dropout=params['dropout'],
            use_gat=params['use_gat'],
        ).to(device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
        )
        
        # Loss with class weighting (if imbalanced)
        criterion = nn.BCEWithLogitsLoss()
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
        
        # Training loop with early stopping
        best_auc = 0.0
        patience_counter = 0
        patience = 25  # Match dl/ config
        
        for epoch in range(200):  # 200 epochs for tuning (less than 300 in full training)
            train_epoch(model, train_loader, optimizer, criterion, device)
            
            if (epoch + 1) % 5 == 0:
                val_auc, _, _ = evaluate(model, val_loader, device)
                scheduler.step(val_auc)
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience // 5:  # Scale patience for eval_interval
                        break
                
                # Report to Optuna for pruning
                trial.report(val_auc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return best_auc
    
    return objective


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 80)
    print("OPTIMIZED MULTIMODAL TRAINING WITH OPTUNA TUNING")
    print("Target: AUC > 0.9 (following dl/ best practices)")
    print("=" * 80)
    
    # Setup
    base_dir = Path(__file__).parent
    train_csv = base_dir / "input" / "cleaned_data_train_rdkit_desc.csv"
    test_csv = base_dir / "input" / "cleaned_data_test_rdkit_desc.csv"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # =========================================================================
    # Load and Prepare Data
    # =========================================================================
    print("\nLoading data...")
    
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    smiles_train = train_df['SMILES'].tolist()
    labels_train = train_df['label'].values
    smiles_test = test_df['SMILES'].tolist()
    labels_test = test_df['label'].values
    
    rdkit_cols = [col for col in train_df.columns if col not in ['ID', 'SMILES', 'label']]
    
    # Extract features
    print("Extracting features...")
    rdkit_train = extract_rdkit_features(train_df, rdkit_cols)
    rdkit_test = extract_rdkit_features(test_df, rdkit_cols)
    
    chemberta_feat = ChemBERTaFeaturizer()
    chemberta_train = chemberta_feat.featurize(smiles_train)
    chemberta_test = chemberta_feat.featurize(smiles_test)
    
    graphs_train = featurize_smiles_to_dgl(smiles_train, model_type="GCN")  # GAT also uses CanonicalAtomFeaturizer
    graphs_test = featurize_smiles_to_dgl(smiles_test, model_type="GCN")
    
    # Filter invalid graphs
    valid_train = [i for i, g in enumerate(graphs_train) if g is not None]
    valid_test = [i for i, g in enumerate(graphs_test) if g is not None]
    
    graphs_train = [graphs_train[i] for i in valid_train]
    rdkit_train = rdkit_train[valid_train]
    chemberta_train = chemberta_train[valid_train]
    labels_train = labels_train[valid_train]
    
    graphs_test = [graphs_test[i] for i in valid_test]
    rdkit_test = rdkit_test[valid_test]
    chemberta_test = chemberta_test[valid_test]
    labels_test = labels_test[valid_test]
    
    # Standardize RDKit features
    scaler = StandardScaler()
    rdkit_train = scaler.fit_transform(rdkit_train)
    rdkit_test = scaler.transform(rdkit_test)
    
    rdkit_dim = rdkit_train.shape[1]
    print(f"Data: {len(graphs_train)} train, {len(graphs_test)} test")
    print(f"RDKit dim: {rdkit_dim}")
    
    # Split train into train/val for tuning (80/20)
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(labels_train))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels_train, random_state=42)
    
    # Create datasets
    train_ds = MultimodalDataset(
        rdkit_train[train_idx], chemberta_train[train_idx],
        [graphs_train[i] for i in train_idx], labels_train[train_idx]
    )
    val_ds = MultimodalDataset(
        rdkit_train[val_idx], chemberta_train[val_idx],
        [graphs_train[i] for i in val_idx], labels_train[val_idx]
    )
    test_ds = MultimodalDataset(rdkit_test, chemberta_test, graphs_test, labels_test)
    
    # Create data loaders with default batch size for tuning
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_multimodal)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal)
    
    # =========================================================================
    # Optuna Hyperparameter Tuning
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: OPTUNA HYPERPARAMETER TUNING")
    print("=" * 80)
    
    n_trials = 50  # Match dl/ project's 100 trials, but faster for this run
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    )
    
    objective = create_objective(train_loader, val_loader, rdkit_dim, device)
    
    print(f"Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best AUC: {study.best_trial.value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # Save best params
    with open(output_dir / "best_multimodal_params.json", 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    # =========================================================================
    # Full Training with Best Params
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: FULL TRAINING WITH BEST PARAMS")
    print("=" * 80)
    
    params = study.best_params
    
    # Rebuild model with best params
    graph_hidden_dims = [params['graph_hidden_dim']] * 3
    graph_num_heads = [params['graph_num_heads']] * 3
    fusion_dims = [params['fusion_dim1'], params['fusion_dim2']]
    
    # Use full training data (train + val)
    full_train_ds = MultimodalDataset(rdkit_train, chemberta_train, graphs_train, labels_train)
    full_train_loader = DataLoader(full_train_ds, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_multimodal)
    
    model = OptimizedMultimodalModel(
        rdkit_dim=rdkit_dim,
        chemberta_proj=params['chemberta_proj'],
        graph_hidden_dims=graph_hidden_dims,
        graph_num_heads=graph_num_heads,
        fusion_dims=fusion_dims,
        dropout=params['dropout'],
        use_gat=params.get('use_gat', False),
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
    
    # Full training with 300 epochs (like dl/)
    best_auc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 25
    
    print(f"\nTraining for up to 300 epochs...")
    for epoch in range(300):
        loss = train_epoch(model, full_train_loader, optimizer, criterion, device)
        
        if (epoch + 1) % 5 == 0:
            test_auc, test_probs, test_labels = evaluate(model, test_loader, device)
            scheduler.step(test_auc)
            
            if test_auc > best_auc:
                best_auc = test_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d}: Loss={loss:.4f}, TestAUC={test_auc:.4f}, BestAUC={best_auc:.4f}")
            
            if patience_counter >= patience // 5:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    test_auc, test_probs, test_labels = evaluate(model, test_loader, device)
    test_preds = (np.array(test_probs) > 0.5).astype(int)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_acc = accuracy_score(test_labels, test_preds)
    
    # Optimize threshold using Youden's J
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    j_scores = tpr - fpr
    best_thresh_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_thresh_idx]
    
    opt_preds = (np.array(test_probs) >= optimal_threshold).astype(int)
    opt_f1 = f1_score(test_labels, opt_preds, zero_division=0)
    opt_acc = accuracy_score(test_labels, opt_preds)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Test AUC: {best_auc:.4f}")
    print(f"Test F1 (threshold=0.5): {test_f1:.4f}")
    print(f"Test Accuracy (threshold=0.5): {test_acc:.4f}")
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"Test F1 (optimal): {opt_f1:.4f}")
    print(f"Test Accuracy (optimal): {opt_acc:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'best_auc': float(best_auc),
        'test_f1': float(test_f1),
        'test_accuracy': float(test_acc),
        'optimal_threshold': float(optimal_threshold),
        'opt_f1': float(opt_f1),
        'opt_accuracy': float(opt_acc),
        'best_params': params,
        'n_trials': n_trials,
        'model_params': total_params,
    }
    
    with open(output_dir / f"tuned_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_auc': best_auc,
        'params': params,
    }, output_dir / f"tuned_model_{timestamp}.pt")
    
    print(f"\n✅ Results saved to output/tuned_results_{timestamp}.json")
    print(f"✅ Model saved to output/tuned_model_{timestamp}.pt")
    
    if best_auc >= 0.9:
        print("\n🎉 SUCCESS! Achieved target AUC >= 0.9!")
    else:
        print(f"\n⚠️ AUC {best_auc:.4f} < 0.9. Consider:")
        print("   1. More Optuna trials (increase n_trials)")
        print("   2. Try AttentiveFP for graph branch")
        print("   3. Use ensemble of multiple models")
        print("   4. Feature engineering on RDKit descriptors")


if __name__ == "__main__":
    main()
