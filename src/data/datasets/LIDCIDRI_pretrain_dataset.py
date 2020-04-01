import torch
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.datasets import BaseDataset
from src.data.transforms import Compose, ToTensor


class LIDCIDRIPretrainDataset(BaseDataset):
    """The dataset of the LIDC-IDRI for self-supervised learning.

    Args:
        data_split_file_path (str):
        preprocess (BoxList): The preprocessing techniques applied to the data.
        transforms (BoxList): 
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
        ct = nii_img.get_fdata().astype(np.float32)[..., np.newaxis]
        ct, = self.preprocess(ct)
        
        transformed_ct, = self.transforms(ct)
        transformed_ct, ct = self.to_tensor(transformed_ct, ct, dtypes=[torch.float, torch.float])
        metadata = {'input': transformed_ct, 'target': ct}
        return metadata

    def __len__(self):
        return len(self.data_paths)
