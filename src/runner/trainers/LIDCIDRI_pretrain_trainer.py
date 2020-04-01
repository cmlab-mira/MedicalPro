import torch

from src.runner.trainers import BaseTrainer


class LIDCIDRIPretrainTrainer(BaseTrainer):
    """The ACDC trainer for the segmentation task.
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
