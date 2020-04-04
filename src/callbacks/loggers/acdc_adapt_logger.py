from src.callbacks.loggers import BaseLogger


class AcdcAdaptLogger(BaseLogger):
    """The ACDC logger for the self-supervised learning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output):
        pass
