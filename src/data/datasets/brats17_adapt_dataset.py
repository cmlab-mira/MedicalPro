import torch
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.datasets import BaseDataset
from src.data.transforms import Compose, ToTensor


class Brats17AdaptDataset(BaseDataset):
    """The dataset of the Multimodal Brain Tumor Segmentation Challenge in MICCAI 2017 (BraTS17)
    for the self-supervised learning.

    Ref:
        https://www.med.upenn.edu/sbia/brats2017.html

    Args:
        data_split_file_path (str): The data split file path.
        preprocess (BoxList): The preprocessing techniques applied to the data.
        transforms (BoxList): The self-supervised transforms applied to the data.
    """

    def __init__(self, data_split_file_path, preprocess, transforms, **kwargs):
        super().__init__(**kwargs)
        if self.type == 'train':
            data_split_file = pd.read_csv(data_split_file_path)
            patient_dirs = map(
                Path,
                data_split_file[
                    (data_split_file.type == 'train') | (data_split_file.type == 'valid')
                ].path
            )
            self.data_paths = tuple(
                (
                    patient_dir / f'{patient_dir.name}_t1.nii.gz',
                    patient_dir / f'{patient_dir.name}_t1ce.nii.gz',
                    patient_dir / f'{patient_dir.name}_t2.nii.gz',
                    patient_dir / f'{patient_dir.name}_flair.nii.gz',
                )
                for patient_dir in patient_dirs
            )
        elif self.type == 'valid':
            self.data_paths = tuple(['nan'] * 4)

        self.preprocess = Compose.compose(preprocess)
        self.transforms = Compose.compose(transforms)
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        t1_path, t1ce_path, t2_path, flair_path = self.data_paths[index]
        t1 = nib.load(t1_path.as_posix()).get_fdata().astype(np.float32)[..., np.newaxis]
        t1ce = nib.load(t1ce_path.as_posix()).get_fdata().astype(np.float32)[..., np.newaxis]
        t2 = nib.load(t2_path.as_posix()).get_fdata().astype(np.float32)[..., np.newaxis]
        flair = nib.load(flair_path.as_posix()).get_fdata().astype(np.float32)[..., np.newaxis]

        t1, t1ce, t2, flair = self.preprocess(t1, t1ce, t2, flair)
        mr = np.concatenate([t1, t1ce, t2, flair], axis=-1)
        transformed_mr, = self.transforms(mr)
        transformed_mr, mr = self.to_tensor(transformed_mr, mr, dtypes=[torch.float, torch.float])
        metadata = {'input': transformed_mr, 'target': mr}
        return metadata

    def __len__(self):
        return len(self.data_paths)
