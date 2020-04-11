import torch

from src.runner.trainers import BaseTrainer


class PretrainTrainer(BaseTrainer):
    """The trainer for for the self-supervised learning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target'].to(self.device)
        output = torch.sigmoid(self.net(input))
        mse_loss = self.loss_fns.mse_loss(output, target)
        loss = self.loss_weights.mse_loss * mse_loss
        return {
            'loss': loss
        }

    def _valid_step(self, batch):
        return self._train_step(batch)


class PretrainMultitaskTrainer(BaseTrainer):
    """The trainer for for the self-supervised learning with domain classification task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_step(self, batch):
        input, target, domain = batch['input'].to(self.device), batch['target'].to(self.device), batch['domain'].to(self.device)
        image_logits, domain_logits = self.net(input)
        output = torch.sigmoid(image_logits)
        mse_loss = self.loss_fns.mse_loss(output, target)
        cross_entropy_loss = self.loss_fns.cross_entropy_loss(domain_logits, domain)
        loss = self.loss_weights.mse_loss * mse_loss + self.loss_weights.cross_entropy_loss * cross_entropy_loss
        return {
            'loss': loss
        }

    def _valid_step(self, batch):
        return self._train_step(batch)