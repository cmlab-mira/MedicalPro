import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F

from src.runner.predictors import BasePredictor


class AcdcSegPredictor(BasePredictor):
    """The ACDC predictor for the segmentation task.
    Args:
        saved_pred (bool): Whether to save the prediction (default: False).
    """

    def __init__(self, saved_pred=False, plot_gamma_performance_curve=False, **kwargs):
        super().__init__(**kwargs)
        if self.test_dataloader.batch_size != 1:
            raise ValueError(f'The testing batch size should be 1. Got {self.test_dataloader.batch_size}.')

        self.saved_pred = saved_pred
        self.plot_gamma_performance_curve = plot_gamma_performance_curve
        self.output_dir = self.saved_dir / 'prediction'
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True)

    def predict(self):
        if self.plot_gamma_performance_curve is False:
            super().predict()
        else:
            state_dict = self.net.state_dict()
            gamma_thresholds = np.arange(0, 1.0, 0.05)
            test_logs = {}
            for gamma_threshold in gamma_thresholds:
                tmp_state_dict = state_dict.copy()
                for key in tmp_state_dict.keys():
                    if ('norm' in key) and ('weight' in key):
                        zeros = torch.zeros_like(tmp_state_dict[key])
                        tmp_state_dict[key] = torch.where(tmp_state_dict[key].abs() <= gamma_threshold, 
                                                          zeros, tmp_state_dict[key])
                self.net.load_state_dict(tmp_state_dict)
                test_log = super().predict()
                if test_logs:
                    for key in test_log.keys():
                        test_logs[key].append(test_log[key])
                else:
                    for key in test_log.keys():
                        test_logs[key] = [test_log[key]]
            for key in test_logs.keys():
                figure_path = self.saved_dir / f'{key}.png'
                fig = plt.figure(figsize=(10, 8))
                plt.plot(gamma_thresholds, test_logs[key])
                plt.xlabel('gamma threshold')
                plt.ylabel(key)
                fig.savefig(figure_path.as_posix())
            
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
            (affine,), (header,), (name,) = batch['affine'], batch['header'], batch['name']
            pred = F.softmax(output, dim=1).argmax(dim=1).squeeze(dim=0).permute(1, 2, 0).contiguous()
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
                'Dice': dice[1:].mean(),
                'DiceRightVentricle': dice[1],
                'DiceMyocardium': dice[2],
                'DiceLeftVentricle': dice[3]
            }
        }
