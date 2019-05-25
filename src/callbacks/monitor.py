import math


class Monitor:
    """The class to monitor the training process and save the model checkpoints.
    Args:
        root (Path): The root directory of the saved model checkpoints.
        mode (str): The mode of the monitor ('max' or 'min').
        target (str): The target of the monitor, usually is loss value or metric score.
        saved_freq (int): The saved frequency.
        early_stop (int): The number early stop.
    """
    def __init__(self, root, mode, target, saved_freq, early_stop):
        self.root = root
        self.mode = mode
        self.target = target
        self.saved_freq = saved_freq
        self.early_stop = math.inf if early_stop == 0 else early_stop
        self.best = -math.inf if self.mode == 'max' else math.inf
        self.not_improved_count = 0

    def is_saved(self, epoch):
        """Whether to save the model checkpoint.
        Args:
            epoch (int): The number of trained epochs.

        Returns:
            path (Path): The path to save the model checkpoint.
        """
        if epoch % self.saved_freq == 0
            return self.root / f'model_{epoch}.pth'
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
            return self.root / 'model_best.pth'
        elif self.mode == 'min' and score < self.best:
            self.best = score
            self.not_improved_count = 0
            return self.root / 'model_best.pth'
        else:
            self.not_improved_count += 1
            return None

    def is_early_stopped(self):
        """Whether to stop the training.
        """
        return self.not_improved_count == self.early_stop
