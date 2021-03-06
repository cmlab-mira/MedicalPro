import torch.nn.functional as F

from src.runner.trainers import BaseTrainer


class AcdcSegTrainer(BaseTrainer):
    """The ACDC trainer for the segmentation task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target'].to(self.device)
        output = self.net(input)
        cross_entropy_loss = self.loss_fns.cross_entropy_loss(output, target.squeeze(dim=1))
        dice_loss = self.loss_fns.dice_loss(output, target)
        loss = (self.loss_weights.cross_entropy_loss * cross_entropy_loss
                + self.loss_weights.dice_loss * dice_loss)
        return {
            'loss': loss,
            'losses': {
                'CrossEntropyLoss': cross_entropy_loss,
                'DiceLoss': dice_loss
            }
        }

    def _valid_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target'].to(self.device)
        output = F.interpolate(self.net(input),
                               size=target.size()[2:],
                               mode='trilinear',
                               align_corners=False)
        cross_entropy_loss = self.loss_fns.cross_entropy_loss(output, target.squeeze(dim=1))
        dice_loss = self.loss_fns.dice_loss(output, target)
        loss = (self.loss_weights.cross_entropy_loss * cross_entropy_loss
                + self.loss_weights.dice_loss * dice_loss)
        dice = self.metric_fns.dice(F.softmax(output, dim=1), target)
        return {
            'loss': loss,
            'losses': {
                'CrossEntropyLoss': cross_entropy_loss,
                'DiceLoss': dice_loss
            },
            'metrics': {
                'Dice': dice[1:].mean(),
                'DiceRightVentricle': dice[1],
                'DiceMyocardium': dice[2],
                'DiceLeftVentricle': dice[3]
            }
        }
