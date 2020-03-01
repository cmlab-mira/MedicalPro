import torch
from torchvision.utils import make_grid

from src.callbacks.loggers.base_logger import BaseLogger


class LUNA16Logger(BaseLogger):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_outputs, valid_batch, valid_outputs):
        """Plot the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_batch (dict): The training batch.
            train_outputs (list of torch.Tensor): The training outputs.
            valid_batch (dict): The validation batch.
            valid_outputs (list of torch.Tensor): The validation outputs.
        """
        train_img = make_grid(train_batch['data'][-1, -1, -1], nrow=8, normalize=True, scale_each=True, pad_value=1)
        valid_img = make_grid(valid_batch['data'][-1, -1, -1], nrow=8, normalize=True, scale_each=True, pad_value=1)

        self.writer.add_image('train', train_img)
        self.writer.add_image('valid', valid_img)