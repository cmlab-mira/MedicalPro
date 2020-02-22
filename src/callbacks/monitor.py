import logging
import math
import sys

LOGGER = logging.getLogger(__name__.split('.')[-1])


class Monitor:
    """The class to monitor the training process and save the model checkpoints.
    Args:
        checkpoints_dir (Path): The root directory of the saved model checkpoints.
        mode (str): The mode of the monitor ('max' or 'min') (default: 'min').
        target (str): The target of the monitor ('loss', 'my_loss' or 'my_metric') (default: 'loss').
        saved_freq (int): The saved frequency (default: 1).
        early_stop (int): The number of times to early stop the training if monitor target is not improved
            (default: 0, do not early stop the training). Notice that the unit is validation times, not epoch.
    """

    def __init__(self, checkpoints_dir, mode='min', target='loss', saved_freq=1, early_stop=0):
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
        score = valid_log.get(self.target)
        if score is None:
            raise KeyError(f"The valid_log has no key named '{self.target}'. "
                           f'Its keys: {list(valid_log.keys())}.\n'
                           'Please check the returned keys as defined in MyTrainer._valid_step().')

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
            'mode': self.mode,
            'target': self.target,
            'saved_freq': self.saved_freq,
            'early_stop': self.early_stop,
            'best': self.best,
            'not_improved_count': self.not_improved_count
        }

    def load_state_dict(self, state_dict):
        if self.mode == state_dict['mode'] and self.target == state_dict['target']:
            self.best = state_dict['best']
            self.not_improved_count = state_dict['not_improved_count']
        else:
            LOGGER.warning(f"The mode and target are changed from "
                           f"{state_dict['mode']} {state_dict['target']} to {self.mode} {self.target}.")

        if self.saved_freq != state_dict['saved_freq']:
            LOGGER.warning(f"The saved_freq is changed from {state_dict['saved_freq']} to {self.saved_freq}.")

        if self.early_stop != state_dict['early_stop']:
            LOGGER.warning(f"The early_stop is changed from {state_dict['early_stop']} to {self.early_stop}.")

        if self.not_improved_count >= self.early_stop:
            LOGGER.critical(f"Load the checkpoint that should have to be early stopped.")
            sys.exit()
