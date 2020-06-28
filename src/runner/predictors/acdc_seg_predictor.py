import torch
import nibabel as nib
import torch.nn.functional as F

from src.runner.predictors import GammaPredictor


class AcdcSegPredictor(GammaPredictor):
    """The ACDC predictor for the segmentation task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _test_step(self, batch):
        if self.test_dataloader.dataset.csv_name == 'testing.csv':
            input, target = batch['input'].to(self.device), batch['target']
            output = F.interpolate(self.net(input),
                                   size=target.size()[2:],
                                   mode='trilinear',
                                   align_corners=False)
            cross_entropy_loss = torch.tensor(float('nan'))
            dice_loss = torch.tensor(float('nan'))
            loss = torch.tensor(float('nan'))
            dice = torch.tensor(tuple(float('nan') for _ in range(4)))
        else:
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

        if self.saved_pred:
            output_dir = self.saved_pred / 'prediction'
            (affine,), (header,), (name,) = batch['affine'], batch['header'], batch['name']
            _, pred = F.softmax(output, dim=1).max(dim=1)
            pred = pred.squeeze(dim=0).permute(1, 2, 0).contiguous()
            nib.save(
                nib.Nifti1Image(
                    pred.cpu().numpy(),
                    affine.numpy(),
                    header
                ),
                (output_dir / name).as_posix()
            )
        return {
            'loss': loss,
            'losses': {
                'CrossEntropyLoss': cross_entropy_loss,
                'DiceLoss': dice_loss
            },
            'metrics': {
                'Dice': dice[1:].mean()
            }
        }
