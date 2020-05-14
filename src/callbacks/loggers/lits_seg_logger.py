from src.callbacks.loggers import BaseLogger


class LitsSegLogger(BaseLogger):
    """The LiTS logger for the segmentation task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output):
        pass
