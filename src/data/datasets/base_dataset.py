from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """The base class for all datasets.
    Args:
        data_dir (Path): The directory of the saved data.
        type_ (str): The type of the dataset ('train', 'valid' or 'test').
    """
    def __init__(self, data_dir, type_):
        super().__init__()
        self.data_dir = data_dir
        self.type = type_
