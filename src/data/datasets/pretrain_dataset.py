import torch
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.datasets import BaseDataset
from src.data.transforms import Compose, ToTensor


class PretrainDataset(BaseDataset):
    """The dataset for self-supervised learning.
    Args:
        data_split_file_path (str): The data split file path.
        preprocess (BoxList): The preprocessing techniques applied to the data.
        transforms (BoxList): The self-supervised transforms applied to the data.
    """

    def __init__(self, data_split_file_path, preprocess, transforms, **kwargs):
        super().__init__(**kwargs)
        data_split_file = pd.read_csv(data_split_file_path)
        self.data_paths = list(map(Path, data_split_file[data_split_file.type == self.type].path))
        self.preprocess = Compose.compose(preprocess)
        self.transforms = Compose.compose(transforms)
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        nii_img = nib.load(data_path.as_posix())
        data = nii_img.get_fdata().astype(np.float32)[..., np.newaxis]
        data, = self.preprocess(data)
        
        transformed_data, = self.transforms(data)
        transformed_data, data = self.to_tensor(transformed_data, data, dtypes=[torch.float, torch.float])
        metadata = {'input': transformed_data, 'target': data}
        return metadata

    def __len__(self):
        return len(self.data_paths)
    

class PretrainMultitaskDataset(BaseDataset):
    """The dataset for self-supervised learning with domain classification task.
    Args:
        data_split_file_path (str): The data split file path.
        preprocess (BoxList): The preprocessing techniques applied to the data.
        transforms (BoxList): The self-supervised transforms applied to the data.
    """

    def __init__(self, data_split_file_path, preprocess, transforms, **kwargs):
        super().__init__(**kwargs)
        data_split_file = pd.read_csv(data_split_file_path)
        self.data_paths = list(map(Path, data_split_file[data_split_file.type == self.type].path))
        self.preprocess = Compose.compose(preprocess)
        self.transforms = Compose.compose(transforms)
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        nii_img = nib.load(data_path.as_posix())
        data = nii_img.get_fdata().astype(np.float32)[..., np.newaxis]
        data, = self.preprocess(data)
        
        if 'LIDC-IDRI' in data_path.as_posix():
            domain = np.array([0])
        elif 'FastMRI' in data_path.as_posix():
            domain = np.array([0])
        else:
            raise ValueError("Unknown data modality.")
        
        transformed_data, = self.transforms(data)
        transformed_data, data = self.to_tensor(transformed_data, data, dtypes=[torch.float, torch.float])
        domain = torch.LongTensor(domain)
        metadata = {'input': transformed_data, 'target': data, 'domain': domain}
        return metadata

    def __len__(self):
        return len(self.data_paths)