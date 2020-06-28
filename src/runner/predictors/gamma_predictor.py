import logging
import matplotlib.pyplot as plt

from src.runner.predictors import BasePredictor
from src.runner.predictors.utils import clip_gamma, get_gamma_percentange

LOGGER = logging.getLogger(__name__.split('.')[-1])


class GammaPredictor(BasePredictor):
    """The gamma predictor for plotting gamma vs performance figure.
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
        self.plot_gamma_performance_curve = plot_gamma_performance_curve
        gamma_thresholds = set(gamma_thresholds)
        gamma_thresholds.add(0)
        self.gamma_thresholds = sorted(gamma_thresholds)

    def predict(self):
        if self.plot_gamma_performance_curve is False:
            super().predict()
        else:
            output_dir = self.saved_dir / 'gamma_performance'
            if not output_dir.is_dir():
                output_dir.mkdir(parents=True)

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
                plt.axhspan(test_logs[key][0] * 0.98, test_logs[key][0] * 1.02, color='black', alpha=0.1)
                plt.plot(self.gamma_thresholds[1:], test_logs[key][1:], color='blue', marker='o')
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
                figure_path = output_dir / f'{key}.png'
                fig.savefig(figure_path.as_posix())
