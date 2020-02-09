import torch
import torch.nn.functional as F

from src.runner.trainers.base_trainer import BaseTrainer


class Luna16Trainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _train_step(self, batch):
        data, label = batch['data'].to(self.device), batch['label'].to(self.device)
        logits = self.net(data)
        cross_entropy_loss = self.loss_fns.cross_entropy_loss(logits, label.squeeze())
        accuracy = self.metric_fns.accuracy(logits, label)
        return {
            'losses': {
                'cross_entropy_loss': cross_entropy_loss
            },
            'metrics': {
                'accuracy': accuracy
            }
        }
        
    def _valid_step(self, batch):
        data, label = batch['data'].to(self.device), batch['label'].to(self.device)
        logits = self.net(data)
        cross_entropy_loss = self.loss_fns.cross_entropy_loss(logits, label.squeeze())
        accuracy = self.metric_fns.accuracy(logits, label)
        return {
            'losses': {
                'cross_entropy_loss': cross_entropy_loss
            },
            'metrics': {
                'accuracy': accuracy
            }
        }