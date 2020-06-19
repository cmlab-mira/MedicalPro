import torch
import nibabel as nib
import torch.nn.functional as F

from src.runner.predictors import BasePredictor


class Brats17SegPredictor(BasePredictor):
    """The BraTS17 predictor for the segmentation task.
    Args:
        saved_pred (bool): Whether to save the prediction (default: False).
    """

    def __init__(self, saved_pred=False, **kwargs):
        super().__init__(**kwargs)
        if self.test_dataloader.batch_size != 1:
            raise ValueError(f'The testing batch size should be 1. Got {self.test_dataloader.batch_size}.')

        self.saved_pred = saved_pred
        self.output_dir = self.saved_dir / 'prediction'
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True)

    def _test_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target'].to(self.device)
        output = F.interpolate(self.net(input),
                               size=target.size()[2:],
                               mode='trilinear',
                               align_corners=False)
        cross_entropy_loss = self.loss_fns.cross_entropy_loss(output, target.squeeze(dim=1))
        dice_loss = self.loss_fns.dice_loss(output, target)
        loss = (self.loss_weights.cross_entropy_loss * cross_entropy_loss
                + self.loss_weights.dice_loss * dice_loss)

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

        if self.saved_pred:
            (affine,), (header,), (name,) = batch['affine'], batch['header'], batch['name']
            _, pred = F.softmax(output, dim=1).max(dim=1)
            pred = pred.squeeze(dim=0).permute(1, 2, 0).contiguous()
            nib.save(
                nib.Nifti1Image(
                    pred.cpu().numpy(),
                    affine.numpy(),
                    header
                ),
                (self.output_dir / name).as_posix()
            )
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
