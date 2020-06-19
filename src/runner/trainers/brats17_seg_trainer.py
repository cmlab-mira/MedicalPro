import torch
import torch.nn.functional as F

from src.runner.trainers import BaseTrainer


class Brats17SegTrainer(BaseTrainer):
    """The BraTS trainer for the segmentation task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target'].to(self.device)
        output = self.net(input)
        cross_entropy_loss = self.loss_fns.cross_entropy_loss(output, target.squeeze(dim=1))
        dice_loss = self.loss_fns.dice_loss(output, target)
        loss = (self.loss_weights.cross_entropy_loss * cross_entropy_loss +
                self.loss_weights.dice_loss * dice_loss)
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
        loss = (self.loss_weights.cross_entropy_loss * cross_entropy_loss +
                self.loss_weights.dice_loss * dice_loss)

        # Enhancing tumor: label 3
        output_prob = F.softmax(output, dim=1)
        output_enhancing_tumor = self.merge_prob(
            output_prob,
            background_indices=[0, 1, 2],
            foreground_indices=[3]
        )
        dice_enhancing_tumor = self.metric_fns.dice(
            output_enhancing_tumor,
            (target == 3).to(torch.long)
        )[1]
        return {
            'loss': loss,
            'losses': {
                'CrossEntropyLoss': cross_entropy_loss,
                'DiceLoss': dice_loss
            },
            'metrics': {
                'DiceEnhancingTumor': dice_enhancing_tumor
            }
        }

    @staticmethod
    def merge_prob(prob, background_indices, foreground_indices):
        background_indices = torch.as_tensor(background_indices, dtype=torch.long, device=prob.device)
        foreground_indices = torch.as_tensor(foreground_indices, dtype=torch.long, device=prob.device)
        background_prob = prob.index_select(dim=1, index=background_indices).sum(dim=1, keepdim=True)
        foreground_prob = prob.index_select(dim=1, index=foreground_indices).sum(dim=1, keepdim=True)
        merged_prob = torch.cat([background_prob, foreground_prob], dim=1)
        return merged_prob
