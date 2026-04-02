"""Training script for multimodal fusion model.

This module provides the training loop and evaluation logic.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple
from pathlib import Path

from .models import MultimodalFusionModule
from .config import TRAINING_PARAMS, OUTPUT_DIR


class MultimodalTrainer:
    """Trainer for multimodal fusion model."""
    
    def __init__(
        self,
        model: MultimodalFusionModule,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        """
        Initialize trainer.
        
        Args:
            model: MultimodalFusionModule instance
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()  # For binary classification
        
    def prepare_batch(self, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """
        Prepare batch data for model input.
        
        Args:
            batch_data: Dictionary with 'rdkit', 'chemberta', 'graph_batch', 'graph_feats' keys
            
        Returns:
            Dictionary with tensors moved to device
        """
        return {
            'rdkit': batch_data['rdkit'].to(self.device),
            'chemberta': batch_data['chemberta'].to(self.device),
            'graph_batch': batch_data['graph_batch'].to(self.device),
            'graph_feats': batch_data['graph_feats'].to(self.device),
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Prepare inputs
            inputs = self.prepare_batch(inputs)
            labels = labels.to(self.device).float()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = self.prepare_batch(inputs)
                labels = labels.to(self.device).float()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                
                total_loss += loss.item()
                
                # Collect predictions
                preds = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'roc_auc': roc_auc_score(all_labels, all_preds),
            'accuracy': accuracy_score(all_labels, all_preds > 0.5),
        }
        
        return total_loss / len(val_loader), metrics
    
    def save_model(self, save_path: Path):
        """Save model checkpoint."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        
    def load_model(self, load_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_multimodal_model(
    train_dataset,
    val_dataset,
    model_config: Dict,
    training_params: Dict = None,
    output_dir: Path = OUTPUT_DIR,
) -> MultimodalTrainer:
    """
    Train multimodal fusion model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_config: Model configuration
        training_params: Training parameters (uses defaults if None)
        output_dir: Output directory for checkpoints
        
    Returns:
        Trained MultimodalTrainer instance
    """
    if training_params is None:
        training_params = TRAINING_PARAMS
    
    # Create model
    model = MultimodalFusionModule(**model_config)
    
    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        learning_rate=training_params['learning_rate'],
        weight_decay=training_params.get('weight_decay', 1e-4),
    )
    
    print(f"\nModel architecture:")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Note: Full training loop would require proper DataLoader setup
    # This is a simplified version for demonstration
    print("\n⚠️  Note: Full training loop requires DataLoader implementation")
    print("    Current implementation provides the trainer instance for manual training")
    print(f"\nTraining configuration:")
    print(f"  - Learning rate: {training_params['learning_rate']}")
    print(f"  - Batch size: {training_params.get('batch_size', 64)}")
    print(f"  - Epochs: {training_params.get('nb_epoch', 100)}")
    print(f"  - Output dir: {output_dir}")
    
    return trainer
