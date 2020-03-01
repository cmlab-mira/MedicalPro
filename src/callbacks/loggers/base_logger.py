from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    """The base class for all loggers.
    Args:
        log_dir (Path): The saved directory.
        net (BaseNet): The network architecture.
        dummy_input (torch.Tensor): The dummy input for plotting the network architecture.
    """

    def __init__(self, log_dir, net, dummy_input=None):
        """
        # TODO: Plot the network architecture.
        # There are some errors: ONNX runtime errors.
        with SummaryWriter(log_dir) as w:
            w.add_graph(net, dummy_input)
        """
        self.writer = SummaryWriter(log_dir)

    def write(self, epoch, train_log, train_batch, train_outputs,
              valid_log=None, valid_batch=None, valid_outputs=None):
        """Plot the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_log (dict): The training log information.
            train_batch (dict or sequence): The training batch.
            train_outputs (torch.Tensor or sequence of torch.Tensor): The training outputs.
            valid_log (dict): The validation log information.
            valid_batch (dict or sequence): The validation batch.
            valid_outputs (torch.Tensor or sequence of torch.Tensor): The validation outputs.
        """
        self._add_scalars(epoch, train_log, valid_log)
        self._add_images(epoch, train_batch, train_outputs, valid_batch, valid_outputs)

    def close(self):
        """Close the writer.
        """
        self.writer.close()

    def _add_scalars(self, epoch, train_log, valid_log=None):
        """Plot the training curves.
        Args:
            epoch (int): The number of trained epochs.
            train_log (dict): The training log information.
            valid_log (dict): The validation log information.
        """
        if valid_log is None:
            for key in train_log.keys():
                self.writer.add_scalars(key, {'train': train_log[key]}, epoch)
        else:
            for key in (set(train_log.keys()) | set(valid_log.keys())):
                scalars = {'train': train_log.get(key), 'valid': valid_log.get(key)}
                scalars = {key: value for key, value in scalars.items() if value is not None}
                self.writer.add_scalars(key, scalars, epoch)

    def _add_images(self, epoch, train_batch, train_outputs, valid_batch, valid_outputs):
        """Plot the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_batch (dict or sequence): The training batch.
            train_outputs (torch.Tensor or sequence of torch.Tensor): The training outputs.
            valid_batch (dict or sequence): The validation batch.
            valid_outputs (torch.Tensor or sequence of torch.Tensor): The validation outputs.
        """
        raise NotImplementedError
