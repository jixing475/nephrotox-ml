"""Early stopping callback for DeepChem models."""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score
import deepchem as dc


class EarlyStoppingCallback:
    """Early stopping callback for DeepChem models.
    
    Monitors validation metric and stops training when no improvement is observed
    for a specified number of epochs (patience).
    
    Attributes:
        val_dataset: Validation dataset for monitoring
        y_val: True labels for validation set
        patience: Number of epochs to wait before stopping
        metric: Metric to monitor ("roc_auc" or "loss")
        eval_interval: Evaluate every N epochs
        save_dir: Directory to save best model (if save_best_model=True)
        save_best_model: Whether to save best model
        best_score: Best metric score observed
        best_epoch: Epoch with best score
        patience_counter: Counter for patience mechanism
        should_stop: Whether training should stop
    """
    
    def __init__(
        self,
        val_dataset: dc.data.Dataset,
        y_val: np.ndarray,
        patience: int = 10,
        metric: str = "roc_auc",
        eval_interval: int = 5,
        save_dir: Optional[pathlib.Path] = None,
        save_best_model: bool = True,
    ):
        """Initialize early stopping callback.
        
        Args:
            val_dataset: Validation dataset
            y_val: True labels for validation set
            patience: Number of epochs without improvement before stopping
            metric: Metric to monitor ("roc_auc" for maximize, "loss" for minimize)
            eval_interval: Evaluate every N epochs
            save_dir: Directory to save best model
            save_best_model: Whether to save best model
        """
        self.val_dataset = val_dataset
        self.y_val = y_val
        self.patience = patience
        self.metric = metric.lower()
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.save_best_model = save_best_model
        
        # Initialize tracking variables
        self.best_score = None
        self.best_epoch = 0
        self.patience_counter = 0
        self.should_stop = False
        self.current_epoch = 0
        
        # Determine if metric should be maximized or minimized
        self.maximize = self.metric == "roc_auc"
        
        # Create save directory if needed
        if self.save_dir and self.save_best_model:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, model: dc.models.Model, epoch: int) -> bool:
        """Called at the end of each epoch.
        
        Args:
            model: DeepChem model being trained
            epoch: Current epoch number (0-indexed)
            
        Returns:
            True if training should continue, False if should stop
        """
        self.current_epoch = epoch
        
        # Only evaluate at specified intervals
        if epoch % self.eval_interval != 0 and epoch > 0:
            return True
        
        # Evaluate model on validation set
        val_proba_raw = model.predict(self.val_dataset)
        
        # Extract probability for positive class
        if val_proba_raw.shape[1] == 2:
            val_proba = val_proba_raw[:, 1]
        else:
            val_proba = val_proba_raw.flatten()
        
        # Calculate metric
        if self.metric == "roc_auc":
            score = roc_auc_score(self.y_val, val_proba)
        elif self.metric == "loss":
            # Calculate binary cross-entropy loss
            epsilon = 1e-15
            val_proba_clipped = np.clip(val_proba, epsilon, 1 - epsilon)
            loss = -np.mean(
                self.y_val * np.log(val_proba_clipped) +
                (1 - self.y_val) * np.log(1 - val_proba_clipped)
            )
            score = -loss  # Negate for consistent comparison (we maximize)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Check if this is the best score
        is_better = False
        if self.best_score is None:
            is_better = True
        elif self.maximize:
            is_better = score > self.best_score
        else:
            is_better = score > self.best_score  # score is already negated for loss
        
        if is_better:
            # New best score found
            self.best_score = score
            self.best_epoch = epoch
            self.patience_counter = 0
            
            # Save best model if enabled
            if self.save_best_model and self.save_dir:
                best_model_path = self.save_dir / "best_model"
                model.save(best_model_path)
        else:
            # No improvement
            self.patience_counter += 1
            
            # Check if patience exceeded
            if self.patience_counter >= self.patience:
                self.should_stop = True
                return False
        
        return True
    
    def get_best_model_path(self) -> Optional[pathlib.Path]:
        """Get path to best model if saved.
        
        Returns:
            Path to best model or None if not saved
        """
        if self.save_best_model and self.save_dir:
            return self.save_dir / "best_model"
        return None


def fit_with_early_stopping(
    model: dc.models.Model,
    train_dataset: dc.data.Dataset,
    val_dataset: dc.data.Dataset,
    y_val: np.ndarray,
    nb_epoch: int,
    early_stopping: Optional[EarlyStoppingCallback] = None,
) -> dc.models.Model:
    """Train model with early stopping support.
    
    Since DeepChem's fit() method doesn't support callbacks directly,
    we implement a custom training loop that evaluates at intervals
    and checks early stopping conditions.
    
    Args:
        model: DeepChem model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset for early stopping
        y_val: True labels for validation set
        nb_epoch: Maximum number of epochs
        early_stopping: Early stopping callback (optional)
        
    Returns:
        Trained model (best model if early stopping was used)
    """
    if early_stopping is None:
        # No early stopping, use standard fit
        model.fit(train_dataset, nb_epoch=nb_epoch)
        return model
    
    # Custom training loop with early stopping
    # DeepChem's fit() doesn't support per-epoch callbacks, so we train
    # in chunks of eval_interval epochs
    eval_interval = early_stopping.eval_interval
    epoch = 0
    
    while epoch < nb_epoch and not early_stopping.should_stop:
        # Train for eval_interval epochs (or remaining epochs)
        epochs_to_train = min(eval_interval, nb_epoch - epoch)
        model.fit(train_dataset, nb_epoch=epochs_to_train)
        epoch += epochs_to_train
        
        # Check early stopping at evaluation intervals
        # Note: epoch is now the total number of epochs trained (1-indexed conceptually)
        # We evaluate when epoch is a multiple of eval_interval or when we've reached max epochs
        if epoch % eval_interval == 0 or epoch >= nb_epoch:
            # Pass epoch - 1 as 0-indexed epoch number
            should_continue = early_stopping.on_epoch_end(model, epoch - 1)
            if not should_continue:
                break
    
    # Load best model if saved
    if early_stopping.save_best_model and early_stopping.get_best_model_path():
        best_model_path = early_stopping.get_best_model_path()
        if best_model_path.exists():
            model.restore(best_model_path)
    
    return model

