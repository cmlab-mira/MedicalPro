from torchvision.utils import make_grid

from src.callbacks.loggers import BaseLogger


class LIDCIDRIPretrainLogger(BaseLogger):
    """The logger for the model pretraining on LIDC-ICRI dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output):
        pass