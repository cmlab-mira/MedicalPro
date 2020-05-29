from src.runner.trainers import BaseTrainer


class LitsAdaptTrainer(BaseTrainer):
    """The LiTS trainer for the self-supervised learning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.monitor.valid_freq <= self.num_epochs:
            raise ValueError(f'The valid_freq={self.monitor.valid_freq} '
                             f'should be greater than num_epoch={self.num_epochs}.')

    def _train_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target'].to(self.device)
        output = self.net(input)
        mse_loss = self.loss_fns.mse_loss(output, target)
        loss = self.loss_weights.mse_loss * mse_loss
        return {
            'loss': loss
        }

    def _valid_step(self, batch):
        pass
