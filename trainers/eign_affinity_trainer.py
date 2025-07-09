#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import exp, pi, cos, log
import torch
from .affinity_trainer import AffinityTrainer


class EIGNAffinityTrainer(AffinityTrainer):
    """
    EIGN-specific trainer with optimized parameters based on original EIGN implementation
    """

    def get_optimizer(self):
        # Use EIGN's original optimizer settings: Adam with weight_decay=1e-6
        weight_decay = getattr(self.config, 'weight_decay', 1e-6)
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.lr, 
            weight_decay=weight_decay
        )
        return optimizer

    def get_scheduler(self, optimizer):
        # Use exponential decay as in original EIGN
        log_alpha = self.log_alpha
        lr_lambda = lambda step: exp(log_alpha * (step + 1))  # equal to alpha^{step}
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'scheduler': scheduler,
            'frequency': 'batch'
        }
