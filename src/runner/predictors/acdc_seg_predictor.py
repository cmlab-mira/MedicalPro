import logging
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F

from src.runner.predictors import BasePredictor
from src.runner.predictors.utils import clip_gamma, get_gamma_percentange

LOGGER = logging.getLogger(__name__.split('.')[-1])


class AcdcSegPredictor(BasePredictor):
    """The ACDC predictor for the segmentation task.
    Args:
        saved_pred (bool): Whether to save the prediction (default: False).
        plot_gamma_performance_curve (bool): Whether to plot gamma vs performance figure (default: False).
        gamma_thresholds (sequence): The thresholds to clip gamma value (default: None).
            Note that this argument is only valid when plot_gamma_performance_curve is True.
    """

    def __init__(self, saved_pred=False, plot_gamma_performance_curve=False, gamma_thresholds=None, **kwargs):
        super().__init__(**kwargs)
        if self.test_dataloader.batch_size != 1:
            raise ValueError(f'The testing batch size should be 1. Got {self.test_dataloader.batch_size}.')

        self.saved_pred = saved_pred
        self.output_dir = self.saved_dir / 'prediction'
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True)
        self.plot_gamma_performance_curve = plot_gamma_performance_curve
        gamma_thresholds = set(gamma_thresholds)
        gamma_thresholds.add(0)
        self.gamma_thresholds = sorted(gamma_thresholds)

    def predict(self):
        if self.plot_gamma_performance_curve is False:
            super().predict()
        else:
            state_dict = self.net.state_dict()
            test_logs = {}
            percentages = []
            for gamma_threshold in self.gamma_thresholds:
                tmp_state_dict = clip_gamma(state_dict, gamma_threshold)
                self.net.load_state_dict(tmp_state_dict)
                print()
                LOGGER.info(f'Gamma threshold: {gamma_threshold}')
                test_log = super().predict()
                if test_logs:
                    for key in test_log.keys():
                        test_logs[key].append(test_log[key])
                else:
                    for key in test_log.keys():
                        test_logs[key] = [test_log[key]]
                percentage = get_gamma_percentange(state_dict, gamma_threshold)
                percentages.append(f'{percentage: .2f}%')
            for key in test_logs.keys():
                fig = plt.figure(figsize=(10, 8))
                plt.axhline(y=test_logs[key][0], label='No cliping', color='black', linestyle='--')
                plt.plot(self.gamma_thresholds[1:], test_logs[key][1:], color='blue', marker='o')
                plt.xscale('log')
                plt.yscale('log')
                for i, percentage in enumerate(percentages[1:], start=1):
                    plt.annotate(
                        percentage,
                        xy=(self.gamma_thresholds[i], test_logs[key][i]),
                        xytext=(-15, -15),
                        textcoords='offset points',
                        ha='center'
                    )
                plt.xlabel('Gamma threshold')
                plt.ylabel(key)
                plt.legend(loc='lower left')
                figure_path = self.saved_dir / f'{key}.png'
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
            loss = (self.loss_weights.cross_entropy_loss * cross_entropy_loss +
                    self.loss_weights.dice_loss * dice_loss)
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
