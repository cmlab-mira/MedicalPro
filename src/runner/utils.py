__all__ = [
    'EpochLog',
]


class EpochLog:
    """The log to record the information of an epoch.
    """

    def __init__(self):
        self.count = 0
        self.log = None

    def _init_log(self, losses=None, metrics=None):
        """Initialize the log.
        Args:
            losses (dict, optional): The computed losses (default: None).
            metrics (dict, optional): The computed metrics (default: None).
        """
        self.log = {}
        self.log['loss'] = 0
        if losses is not None:
            for loss_name in losses.keys():
                self.log[loss_name] = 0
        if metrics is not None:
            for metric_name in metrics.keys():
                self.log[metric_name] = 0

    def update(self, batch_size, loss, losses=None, metrics=None):
        """Accumulate the computed losses and metrics.
        Args:
            batch_size (int): The batch size.
            loss (torch.Tensor): The computed loss.
            losses (dict, optional): The computed losses (default: None).
            metrics (dict, optional): The computed metrics (default: None).
        """
        self.count += batch_size
        if self.log is None:
            self._init_log(losses, metrics)
        self.log['loss'] += loss.item() * batch_size
        if losses is not None:
            for loss_name, loss in losses.items():
                self.log[loss_name] += loss.item() * batch_size
        if metrics is not None:
            for metric_name, metric in metrics.items():
                self.log[metric_name] += metric.item() * batch_size

    @property
    def on_step_end_log(self):
        return dict((key, f'{value / self.count: .3f}') for key, value in self.log.items())

    @property
    def on_epoch_end_log(self):
        return dict((key, value / self.count) for key, value in self.log.items())
