import math

__all__ = ['Monitor']


class Monitor:
    """The class to monitor the training process and save the model checkpoints.
    Args:
        checkpoints_dir (Path): The root directory of the saved model checkpoints.
        mode (str): The mode of the monitor ('max' or 'min').
        target (str): The target of the monitor ('loss', 'my_loss' or 'my_metric').
        saved_freq (int): The saved frequency.
        early_stop (int): The number of times to early stop the training if monitor target is not improved
            (default: 0, do not early stop the training). Notice that the unit is validation times, not epoch.
    """

    def __init__(self, checkpoints_dir, mode, target, saved_freq, early_stop=0):
        self.checkpoints_dir = checkpoints_dir
        if mode not in ['min', 'max']:
            raise ValueError(f"The mode should be 'min' or 'max'. Got {mode}.")
        self.mode = mode
        self.target = target
        self.saved_freq = saved_freq
        self.early_stop = math.inf if early_stop == 0 else early_stop
        self.best = -math.inf if self.mode == 'max' else math.inf
        self.not_improved_count = 0

        if not self.checkpoints_dir.is_dir():
            self.checkpoints_dir.mkdir(parents=True)

    def is_saved(self, epoch):
        """Whether to save the model checkpoint.
        Args:
            epoch (int): The number of trained epochs.

        Returns:
            path (Path): The path to save the model checkpoint.
        """
        if epoch % self.saved_freq == 0:
            return self.checkpoints_dir / f'model_{epoch}.pth'
        else:
            return None

    def is_best(self, valid_log):
        """Whether to save the best model checkpoint.
        Args:
            valid_log (dict): The validation log information.

        Returns:
            path (Path): The path to save the model checkpoint.
        """
        score = valid_log[self.target]
        if self.mode == 'max' and score > self.best:
            self.best = score
            self.not_improved_count = 0
            return self.checkpoints_dir / 'model_best.pth'
        elif self.mode == 'min' and score < self.best:
            self.best = score
            self.not_improved_count = 0
            return self.checkpoints_dir / 'model_best.pth'
        else:
            self.not_improved_count += 1
            return None

    def is_early_stopped(self):
        """Whether to early stop the training.
        """
        return self.not_improved_count == self.early_stop

    def state_dict(self):
        return {
            'checkpoints_dir': self.checkpoints_dir,
            'mode': self.mode,
            'target': self.target,
            'saved_freq': self.saved_freq,
            'early_stop': self.early_stop,
            'best': self.best,
            'not_improved_count': self.not_improved_count
        }

    def load_state_dict(self, state_dict):
        self.checkpoints_dir = state_dict['checkpoints_dir']
        self.mode = state_dict['mode']
        self.target = state_dict['target']
        self.saved_freq = state_dict['saved_freq']
        self.early_stop = state_dict['early_stop']
        self.best = state_dict['best']
        self.not_improved_count = state_dict['not_improved_count']
