from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """The base class for all datasets.
    Args:
        type_ (str): The type of the dataset ('train', 'valid' or 'test').
    """

    def __init__(self, type_):
        super().__init__()
        self.type = type_
