import torch
from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    """The base class for all loggers.
    Args:
        log_dir (str): The saved directory.
        net (BaseNet): The network architecture.
        dummy_input (torch.Tensor): The dummy input for plotting the network architecture.
    """
    def __init__(self, log_dir, net, dummy_input):
        # Plot the network architecture.
        with SummaryWriter(log_dir) as w:
            w.add_graph(net, dummy_input)

        self.writer = SummaryWriter(log_dir)

    def write(self, epoch, train_log, train_batch, train_output, valid_log, valid_batch, valid_output):
        """Plot the network architecture and the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_log (dict): The training log information.
            train_batch (dict): The training batch.
            train_output (torch.Tensor): The training output.
            valid_log (dict): The validation log information.
            valid_batch (dict): The validation batch.
            valid_output (torch.Tensor): The validation output.
        """
        self._add_scalars(epoch, train_log, valid_log)
        self._add_images(train_batch, train_output, valid_batch, valid_output)

    def _add_scalars(self, epoch, train_log, valid_log):
        """Plot the training curves.
        Args:
            epoch (int): The number of trained epochs.
            train_log (dict): The training log information.
            valid_log (dict): The validation log information.
        """
        for key in train_log:
            self.writer.add_scalars(key, {'train': train_log[key], 'valid': valid_log[key]}, epoch)

    def _add_images(self, train_batch, train_output, valid_batch, valid_output):
        """Plot the visualization results.
        Args:
            train_batch (dict): The training batch.
            train_output (torch.Tensor): The training output.
            valid_batch (dict): The validation batch.
            valid_output (torch.Tensor): The validation output.
        """
        raise NotImplementedError
