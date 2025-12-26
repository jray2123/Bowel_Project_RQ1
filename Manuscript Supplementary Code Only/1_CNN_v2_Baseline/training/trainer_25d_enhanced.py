#!/usr/bin/env python3
"""
Enhanced 2.5D Trainer for Multi-Organ Trauma Detection

This module implements the enhanced trainer following the Enhanced 2.5D Baseline 
Execution Plan specifications for Phase 3: Training Infrastructure.

Key features:
- Layer-wise learning rate optimization for ViT
- Enhanced loss function with uncertainty estimation
- AdamW optimizer with cosine annealing and warm restarts
- Gradient clipping and mixed precision training
- Early stopping based on bowel sensitivity
- Memory-optimized for RTX 3090 constraints

Training specifications from execution plan:
- CNN: LR=1e-4, batch=4, grad_accum=4, bowel_weight=3.0
- ViT: LR=2e-4, batch=6, grad_accum=3, bowel_weight=3.5, layer-wise LR
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from tqdm import tqdm
import warnings
import math

from training.losses import MultiOrganLoss
from models.cnn_25d_enhanced import CNN25DEnhanced
from models.vit_25d_enhanced import ViT25DEnhanced
from models.swin_25d_enhanced import SwinTransformer25DEnhanced


class Enhanced25DTrainer:
    """
    Enhanced trainer for 2.5D multi-organ models following execution plan.
    
    Supports CNN25DEnhanced, ViT25DEnhanced, and SwinTransformer25DEnhanced with:
    - Layer-wise learning rates (ViT)
    - Enhanced optimization strategies
    - Memory-efficient training
    - Advanced scheduling and early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        experiment_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Enhanced 2.5D Trainer.
        
        Args:
            model: CNN25DEnhanced or ViT25DEnhanced model
            train_loader: Training data loader
            val_loader: Validation data loader  
            config: Training configuration dictionary
            device: Training device
            experiment_dir: Directory for saving outputs
            logger: Optional logger instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.logger = logger or self._setup_logger()
        
        # Extract training config
        self.train_config = config.get('training', {})
        self.loss_config = config.get('loss', {})
        
        # Training parameters from execution plan
        self.max_epochs = self.train_config.get('max_epochs', 25)
        self.learning_rate = self.train_config.get('learning_rate', 1e-4)
        self.weight_decay = self.train_config.get('weight_decay', 1e-5)
        self.grad_clip_norm = self.train_config.get('grad_clip_norm', 1.0)
        self.gradient_accumulation_steps = self.train_config.get('gradient_accumulation_steps', 4)
        self.use_amp = self.train_config.get('use_amp', True)
        self.warmup_epochs = self.train_config.get('warmup_epochs', 3)
        self.patience = self.train_config.get('patience', 8)
        self.log_interval = self.train_config.get('log_interval', 50)
        
        # Model-specific settings
        self.is_vit = isinstance(model, ViT25DEnhanced)
        self.is_swin = isinstance(model, SwinTransformer25DEnhanced)
        self.layer_wise_lr = self.train_config.get('layer_wise_lr_optimization', False) and (self.is_vit or self.is_swin)
        
        # Initialize training components
        self._setup_loss_function()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_amp()
        
        # Training state
        self.current_epoch = 0
        self.best_score = -float('inf')
        self.best_bowel_sensitivity = 0.0
        self.epochs_without_improvement = 0
        self.training_history = []
        
        # Create directories
        self.checkpoints_dir = self.experiment_dir / 'checkpoints'
        self.logs_dir = self.experiment_dir / 'logs'
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Initialized Enhanced 2.5D Trainer:")
        self.logger.info(f"  Model: {type(model).__name__}")
        self.logger.info(f"  Layer-wise LR: {self.layer_wise_lr}")
        self.logger.info(f"  Learning rate: {self.learning_rate}")
        self.logger.info(f"  Batch size: {train_loader.batch_size}")
        self.logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        self.logger.info(f"  Mixed precision: {self.use_amp}")
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for training."""
        logger = logging.getLogger('Enhanced25DTrainer')
        logger.setLevel(logging.INFO)
        
        # Console handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _setup_loss_function(self):
        """Setup multi-organ loss function."""
        # Get organ-specific weights from config
        organ_weights = {
            'bowel': self.loss_config.get('bowel_loss_weight', 3.0),
            'extravasation': self.loss_config.get('other_organ_weights', {}).get('extravasation', 1.0),
            'liver': self.loss_config.get('other_organ_weights', {}).get('liver', 1.0),
            'kidney': self.loss_config.get('other_organ_weights', {}).get('kidney', 1.0),
            'spleen': self.loss_config.get('other_organ_weights', {}).get('spleen', 1.0)
        }
        
        self.loss_function = MultiOrganLoss(
            focal_alpha=self.loss_config.get('focal_alpha', 0.75),
            focal_gamma=self.loss_config.get('focal_gamma', 3.0),
            organ_weights=organ_weights
        ).to(self.device)
        
        self.logger.info(f"Loss function initialized with organ weights: {organ_weights}")
        
    def _setup_optimizer(self):
        """Setup optimizer with optional layer-wise learning rates."""
        if self.layer_wise_lr and self.is_vit:
            # ViT with layer-wise learning rates (execution plan specification)
            param_groups = self.model.get_parameter_groups(base_lr=self.learning_rate)
            self.optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
            
            self.logger.info("Using layer-wise learning rates for ViT:")
            for group in param_groups:
                num_params = sum(p.numel() for p in group['params'])
                self.logger.info(f"  {group['name']}: LR={group['lr']:.1e}, Params={num_params:,}")
                
        else:
            # Standard optimizer for CNN or uniform ViT
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Using standard AdamW: LR={self.learning_rate:.1e}, Params={total_params:,}")
            
    def _setup_scheduler(self):
        """Setup learning rate scheduler with cosine annealing and warm restarts."""
        scheduler_type = self.train_config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            # Cosine annealing with warm restarts (execution plan specification)
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(1, self.max_epochs // 3),  # Restart every 1/3 of training
                T_mult=2,  # Double restart period each time
                eta_min=self.learning_rate * 0.01  # Minimum LR is 1% of initial
            )
        else:
            # Fallback to step scheduler
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.max_epochs // 3,
                gamma=0.1
            )
            
        self.logger.info(f"Scheduler: {scheduler_type} with warmup for {self.warmup_epochs} epochs")
        
    def _setup_amp(self):
        """Setup automatic mixed precision."""
        if self.use_amp:
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            self.logger.info("Mixed precision training disabled")
            
    def _warmup_lr(self, epoch: int, batch_idx: int):
        """Apply learning rate warmup."""
        if epoch >= self.warmup_epochs:
            return
            
        # Linear warmup
        total_warmup_steps = self.warmup_epochs * len(self.train_loader)
        current_step = epoch * len(self.train_loader) + batch_idx
        warmup_factor = min(1.0, current_step / total_warmup_steps)
        
        # Apply warmup to all parameter groups
        for param_group in self.optimizer.param_groups:
            base_lr = param_group.get('initial_lr', param_group['lr'])
            param_group['lr'] = base_lr * warmup_factor
            
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Set model epoch for backbone freezing control
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(self.current_epoch)
            
        epoch_losses = {'total': 0.0, 'bowel': 0.0, 'extravasation': 0.0, 
                       'liver': 0.0, 'kidney': 0.0, 'spleen': 0.0}
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.max_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Warmup learning rate
            self._warmup_lr(self.current_epoch, batch_idx)
            
            # Forward pass
            loss_dict, _ = self._forward_pass(batch, training=True)
            
            # Backward pass with gradient accumulation
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
            else:
                loss_dict['total'].backward()
                
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                
            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}",
                    'bowel': f"{loss_dict['bowel'].item():.4f}",
                    'lr': f"{current_lr:.2e}"
                })
                
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
        
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_losses = {'total': 0.0, 'bowel': 0.0, 'extravasation': 0.0,
                       'liver': 0.0, 'kidney': 0.0, 'spleen': 0.0}
        all_predictions = {organ: [] for organ in epoch_losses.keys() if organ != 'total'}
        all_targets = {organ: [] for organ in epoch_losses.keys() if organ != 'total'}
        num_batches = 0
        
        with torch.no_grad():
            # Use enumerate instead of tqdm to avoid progress bar issues
            for i, batch in enumerate(self.val_loader):
                if i % 10 == 0:  # Print progress every 10 batches
                    print(f"  Validation batch {i+1}/{len(self.val_loader)}")
                    
                batch = batch  # Process batch normally
                # Forward pass (single pass now returns both loss and outputs)
                loss_dict, outputs = self._forward_pass(batch, training=False)
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    if key in epoch_losses:
                        epoch_losses[key] += value.item()
                        
                # Store predictions and targets for metrics
                targets = self._extract_targets(batch)
                
                for organ in all_predictions:
                    if organ in outputs and organ in targets:
                        all_predictions[organ].extend(torch.softmax(outputs[organ], dim=1).cpu().numpy())
                        all_targets[organ].extend(targets[organ].cpu().numpy())
                        
                num_batches += 1
                
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        # Calculate metrics (focusing on bowel sensitivity for early stopping)
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        return {**epoch_losses, **metrics}
        
    def _forward_pass(self, batch: Dict, training: bool = True) -> tuple:
        """Forward pass through model and loss computation."""
        # Move batch to device
        slice_sequence = batch['slice_sequence'].to(self.device)
        position_encoding = batch['position_encoding'].to(self.device)
        sequence_mask = batch['sequence_mask'].to(self.device)
        targets = {organ: labels.to(self.device) for organ, labels in batch['labels'].items()}
        
        # Forward pass through model
        if self.use_amp and training:
            with autocast():
                outputs = self.model(slice_sequence, position_encoding, sequence_mask)
                loss_dict = self.loss_function(outputs, targets)
        else:
            outputs = self.model(slice_sequence, position_encoding, sequence_mask)
            loss_dict = self.loss_function(outputs, targets)
            
        return loss_dict, outputs
        
    def _get_model_outputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Get model outputs for metrics calculation."""
        slice_sequence = batch['slice_sequence'].to(self.device)
        position_encoding = batch['position_encoding'].to(self.device)  
        sequence_mask = batch['sequence_mask'].to(self.device)
        
        return self.model(slice_sequence, position_encoding, sequence_mask)
        
    def _extract_targets(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract targets from batch."""
        return {organ: labels.to(self.device) for organ, labels in batch['labels'].items()}
        
    def _calculate_metrics(self, predictions: Dict, targets: Dict) -> Dict[str, float]:
        """Calculate validation metrics."""
        metrics = {}
        
        # Focus on bowel sensitivity for early stopping (execution plan requirement)
        if 'bowel' in predictions and 'bowel' in targets:
            pred_probs = np.array(predictions['bowel'])
            true_labels = np.array(targets['bowel'])
            
            # Binary classification metrics
            pred_binary = (pred_probs[:, 1] > 0.5).astype(int)
            
            # Sensitivity (recall for positive class)
            true_positives = ((pred_binary == 1) & (true_labels == 1)).sum()
            false_negatives = ((pred_binary == 0) & (true_labels == 1)).sum()
            sensitivity = true_positives / (true_positives + false_negatives + 1e-8)
            
            # Specificity  
            true_negatives = ((pred_binary == 0) & (true_labels == 0)).sum()
            false_positives = ((pred_binary == 1) & (true_labels == 0)).sum()
            specificity = true_negatives / (true_negatives + false_positives + 1e-8)
            
            metrics['bowel_sensitivity'] = float(sensitivity)
            metrics['bowel_specificity'] = float(specificity)
            
        return metrics
        
    def train(self) -> Dict:
        """Full training loop."""
        self.logger.info("Starting enhanced 2.5D training...")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Scheduler step (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
                
            # Logging
            self._log_epoch_results(train_metrics, val_metrics)
            
            # Save checkpoint
            self._save_checkpoint(val_metrics)
            
            # Early stopping check (based on bowel sensitivity per execution plan)
            bowel_sensitivity = val_metrics.get('bowel_sensitivity', 0.0)
            improved = bowel_sensitivity > self.best_bowel_sensitivity
            
            if improved:
                self.best_bowel_sensitivity = bowel_sensitivity
                self.epochs_without_improvement = 0
                self._save_best_model()
            else:
                self.epochs_without_improvement += 1
                
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                self.logger.info(f"Early stopping after {self.patience} epochs without improvement")
                break
                
        self.logger.info("Training completed!")
        return self.training_history
        
    def _log_epoch_results(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch results."""
        # Current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Log to console
        self.logger.info(f"Epoch {self.current_epoch+1}/{self.max_epochs}:")
        self.logger.info(f"  Train Loss: {train_metrics['total']:.4f} (bowel: {train_metrics['bowel']:.4f})")
        self.logger.info(f"  Val Loss: {val_metrics['total']:.4f} (bowel: {val_metrics['bowel']:.4f})")
        self.logger.info(f"  Bowel Sensitivity: {val_metrics.get('bowel_sensitivity', 0.0):.4f}")
        self.logger.info(f"  Learning Rate: {current_lr:.2e}")
        
        # Store in history
        epoch_data = {
            'epoch': self.current_epoch + 1,
            'train_loss': train_metrics['total'],
            'val_loss': val_metrics['total'],
            'train_bowel_loss': train_metrics['bowel'],
            'val_bowel_loss': val_metrics['bowel'],
            'bowel_sensitivity': val_metrics.get('bowel_sensitivity', 0.0),
            'bowel_specificity': val_metrics.get('bowel_specificity', 0.0),
            'learning_rate': current_lr
        }
        
        self.training_history.append(epoch_data)
        
        # Save training history
        history_path = self.logs_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
            
    def _save_checkpoint(self, val_metrics: Dict):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metrics': val_metrics,
            'best_bowel_sensitivity': self.best_bowel_sensitivity,
            'training_history': self.training_history,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        # Save latest checkpoint
        checkpoint_path = self.checkpoints_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Always save epoch checkpoint for model selection
        epoch_checkpoint_path = self.checkpoints_dir / f'epoch_{self.current_epoch+1}_checkpoint.pth'
        torch.save(checkpoint, epoch_checkpoint_path)
        
        # Also save periodic checkpoints for storage management
        if (self.current_epoch + 1) % self.train_config.get('save_every_n_epochs', 5) == 0:
            periodic_checkpoint_path = self.checkpoints_dir / f'periodic_epoch_{self.current_epoch+1}_checkpoint.pth'
            torch.save(checkpoint, periodic_checkpoint_path)
            
    def _save_best_model(self):
        """Save best model based on bowel sensitivity."""
        best_model_path = self.checkpoints_dir / 'best_model.pth'
        torch.save({
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'best_bowel_sensitivity': self.best_bowel_sensitivity,
            'config': self.config
        }, best_model_path)
        
        self.logger.info(f"New best model saved with bowel sensitivity: {self.best_bowel_sensitivity:.4f}")


def create_enhanced_25d_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    experiment_dir: str
) -> Enhanced25DTrainer:
    """
    Create Enhanced 2.5D Trainer from configuration.
    
    Args:
        model: CNN25DEnhanced or ViT25DEnhanced model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Training device
        experiment_dir: Experiment directory
        
    Returns:
        Initialized Enhanced25DTrainer
    """
    return Enhanced25DTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_dir=experiment_dir
    )