#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Enhanced Affinity Trainer with ReduceLROnPlateau Scheduler
支持更好的学习率调度和早期停止机制
"""
from math import exp, pi, cos, log
import torch
from trainers.abs_trainer import Trainer


class EnhancedAffinityTrainer(Trainer):
    """Enhanced Affinity Trainer with better learning rate scheduling"""

    def __init__(self, model, train_loader, valid_loader, config):
        self.global_step = 0
        self.epoch = 0
        self.max_step = config.max_epoch * config.step_per_epoch
        self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        
        # Enhanced scheduler parameters
        self.patience = getattr(config, 'patience', 10)  # epochs to wait before reducing lr
        self.factor = getattr(config, 'factor', 0.5)     # factor to multiply lr by
        self.min_lr = getattr(config, 'min_lr', 1e-7)    # minimum learning rate
        self.cooldown = getattr(config, 'cooldown', 5)   # epochs to wait after lr reduction
        
        # Early stopping parameters
        self.early_stopping_patience = getattr(config, 'early_stopping_patience', 20)
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        super().__init__(model, train_loader, valid_loader, config)

    def get_optimizer(self):
        from torch.optim import Adam
        optimizer = Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1e-3)
        return optimizer

    def get_scheduler(self, optimizer):
        # Use ReduceLROnPlateau for better learning rate scheduling
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',                    # minimize validation loss
            factor=self.factor,            # multiply lr by this factor
            patience=self.patience,        # number of epochs to wait
            min_lr=self.min_lr,           # minimum learning rate
            cooldown=self.cooldown        # epochs to wait after lr reduction
        )
        return {
            'scheduler': scheduler,
            'frequency': 'epoch',         # update per epoch, not per batch
            'monitor': 'valid_loss'       # monitor validation loss
        }

    def lr_weight(self, step):
        if self.global_step >= self.config.warmup:
            return 0.99 ** self.epoch
        return (self.global_step + 1) * 1.0 / self.config.warmup

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    def _before_train_epoch_start(self):
        # reform batch, with new random batches
        self.train_loader.dataset._form_batch()
        return super()._before_train_epoch_start()

    def _after_valid_epoch_end(self, val_loss):
        """Handle early stopping and scheduler updates"""
        # Update scheduler with validation loss
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'step'):
                if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
        
        # Early stopping logic
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            
        # Log early stopping info
        self.log('epochs_no_improve', self.epochs_no_improve, 0, True)
        self.log('best_val_loss', self.best_val_loss, 0, True)
        
        # Check if should stop early
        if self.epochs_no_improve >= self.early_stopping_patience:
            print(f"Early stopping triggered after {self.epochs_no_improve} epochs without improvement")
            return True  # Signal to stop training
        
        return False  # Continue training

    def share_step(self, batch, batch_idx, val=False):
        loss = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'])

        log_type = 'Validation' if val else 'Train'

        self.log(f'Loss/{log_type}', loss, batch_idx, val)

        if not val:
            lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = lr[0]
            self.log('lr', lr, batch_idx, val)

        return loss
