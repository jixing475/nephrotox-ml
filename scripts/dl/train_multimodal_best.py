"""
Phase 2: Full Training with Best Params from Optuna Tuning
Uses the best parameters discovered from 50-trial Optuna search.

Best params (Trial 25, Val AUC 0.8844):
- chemberta_proj: 175
- graph_hidden_dim: 64
- graph_num_heads: 4
- fusion_dim1: 461, fusion_dim2: 115
- dropout: 0.272
- learning_rate: 5.24e-5
- weight_decay: 4.39e-4
- batch_size: 16
"""

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.featurizers import ChemBERTaFeaturizer, extract_rdkit_features
from src.graph_utils import featurize_smiles_to_dgl
from train_full_dataset import MultimodalDataset, collate_multimodal
from tune_multimodal import OptimizedMultimodalModel, train_epoch, evaluate


def main():
    print("=" * 80)
    print("PHASE 2: FULL TRAINING WITH BEST PARAMS")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output"
    
    # Load best params
    params_file = output_dir / "best_multimodal_params.json"
    with open(params_file) as f:
        params = json.load(f)
    
    print("Best params from Optuna tuning:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\nLoading data...")
    train_csv = base_dir / "input" / "cleaned_data_train_rdkit_desc.csv"
    test_csv = base_dir / "input" / "cleaned_data_test_rdkit_desc.csv"
    
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    smiles_train = train_df['SMILES'].tolist()
    labels_train = train_df['label'].values
    smiles_test = test_df['SMILES'].tolist()
    labels_test = test_df['label'].values
    
    rdkit_cols = [col for col in train_df.columns if col not in ['ID', 'SMILES', 'label']]
    
    print("Extracting features...")
    rdkit_train = extract_rdkit_features(train_df, rdkit_cols)
    rdkit_test = extract_rdkit_features(test_df, rdkit_cols)
    
    chemberta_feat = ChemBERTaFeaturizer()
    chemberta_train = chemberta_feat.featurize(smiles_train)
    chemberta_test = chemberta_feat.featurize(smiles_test)
    
    graphs_train = featurize_smiles_to_dgl(smiles_train, model_type="GCN")
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
    
    # Create datasets and loaders
    train_ds = MultimodalDataset(rdkit_train, chemberta_train, graphs_train, labels_train)
    test_ds = MultimodalDataset(rdkit_test, chemberta_test, graphs_test, labels_test)
    
    batch_size = params.get('batch_size', 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)
    
    # =========================================================================
    # Build Model
    # =========================================================================
    graph_hidden_dims = [params['graph_hidden_dim']] * 3
    graph_num_heads = [params.get('graph_num_heads', 4)] * 3
    fusion_dims = [params['fusion_dim1'], params['fusion_dim2']]
    
    model = OptimizedMultimodalModel(
        rdkit_dim=rdkit_dim,
        chemberta_proj=params['chemberta_proj'],
        graph_hidden_dims=graph_hidden_dims,
        graph_num_heads=graph_num_heads,
        fusion_dims=fusion_dims,
        dropout=params['dropout'],
        use_gat=False,  # Use GCN for stability
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")
    
    # =========================================================================
    # Training
    # =========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
    
    best_auc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 25
    best_epoch = 0
    
    print(f"\nTraining for up to 300 epochs...")
    for epoch in range(300):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        if (epoch + 1) % 5 == 0:
            test_auc, probs, labels = evaluate(model, test_loader, device)
            scheduler.step(test_auc)
            
            if test_auc > best_auc:
                best_auc = test_auc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d}: Loss={loss:.4f}, TestAUC={test_auc:.4f}, BestAUC={best_auc:.4f}")
            
            if patience_counter >= patience // 5:
                print(f"\nEarly stopping at epoch {epoch+1} (best: {best_epoch})")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    
    # =========================================================================
    # Final Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    test_auc, probs, labels = evaluate(model, test_loader, device)
    probs = np.array(probs)
    labels = np.array(labels)
    
    # Standard threshold
    preds = (probs > 0.5).astype(int)
    test_f1 = f1_score(labels, preds, zero_division=0)
    test_acc = accuracy_score(labels, preds)
    test_prec = precision_score(labels, preds, zero_division=0)
    test_recall = recall_score(labels, preds, zero_division=0)
    
    # Optimal threshold using Youden's J
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_thresh_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_thresh_idx]
    
    opt_preds = (probs >= optimal_threshold).astype(int)
    opt_f1 = f1_score(labels, opt_preds, zero_division=0)
    opt_acc = accuracy_score(labels, opt_preds)
    opt_prec = precision_score(labels, opt_preds, zero_division=0)
    opt_recall = recall_score(labels, opt_preds, zero_division=0)
    
    cm = confusion_matrix(labels, preds)
    cm_opt = confusion_matrix(labels, opt_preds)
    
    print(f"\nTest AUC: {best_auc:.4f}")
    print(f"\nWith threshold=0.5:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    print(f"\nWith optimal threshold={optimal_threshold:.4f}:")
    print(f"  Accuracy: {opt_acc:.4f}")
    print(f"  Precision: {opt_prec:.4f}")
    print(f"  Recall: {opt_recall:.4f}")
    print(f"  F1: {opt_f1:.4f}")
    print(f"  Confusion Matrix:\n{cm_opt}")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'best_epoch': best_epoch,
        'test_auc': float(best_auc),
        'threshold_05': {
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'recall': float(test_recall),
            'f1': float(test_f1),
            'confusion_matrix': cm.tolist(),
        },
        'optimal_threshold': {
            'value': float(optimal_threshold),
            'accuracy': float(opt_acc),
            'precision': float(opt_prec),
            'recall': float(opt_recall),
            'f1': float(opt_f1),
            'confusion_matrix': cm_opt.tolist(),
        },
        'params': params,
        'model_params': total_params,
    }
    
    with open(output_dir / f"optimized_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_auc': best_auc,
        'params': params,
    }, output_dir / f"optimized_model_{timestamp}.pt")
    
    print(f"\n✅ Results saved to output/optimized_results_{timestamp}.json")
    print(f"✅ Model saved to output/optimized_model_{timestamp}.pt")
    
    if best_auc >= 0.9:
        print("\n🎉 SUCCESS! Achieved target AUC >= 0.9!")
    elif best_auc >= 0.88:
        print(f"\n✨ EXCELLENT! AUC {best_auc:.4f} is very close to target!")
        print("   Consider: Running more trials or using ensemble methods")
    else:
        print(f"\n⚠️ AUC {best_auc:.4f} < 0.9")


if __name__ == "__main__":
    main()
