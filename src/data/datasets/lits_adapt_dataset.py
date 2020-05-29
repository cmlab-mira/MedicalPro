import torch
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.datasets import BaseDataset
from src.data.transforms import Compose, ToTensor


class LitsAdaptDataset(BaseDataset):
    """The dataset of the Liver Tumor Segmentation Challenge (LiTS) in MICCAI 2017
    for the self-supervised learning.

    Ref:
        https://competitions.codalab.org/competitions/17094
        https://github.com/PatrickChrist/LITS-CHALLENGE/blob/master/submission-guide.md

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
                data_path
                for patient_dir in patient_dirs
                for data_path in sorted(patient_dir.glob('**/*volume-*.nii'))
            )
        elif self.type == 'valid':
            self.data_paths = tuple(['nan'])

        self.preprocess = Compose.compose(preprocess)
        self.transforms = Compose.compose(transforms)
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        ct_path = self.data_paths[index]
        nii_img = nib.load(ct_path.as_posix())
        ct = nii_img.get_fdata().astype(np.float32)[..., np.newaxis]
        input_spacing = nii_img.header['pixdim'][1:4]
        transforms_kwargs = {
            'Resample': {
                'input_spacings': (input_spacing,),
                'orders': (1,)
            }
        }
        ct, = self.preprocess(ct, **transforms_kwargs)
        transformed_ct, = self.transforms(ct)
        transformed_ct, ct = self.to_tensor(transformed_ct, ct, dtypes=[torch.float, torch.float])
        metadata = {'input': transformed_ct, 'target': ct}
        return metadata

    def __len__(self):
        return len(self.data_paths)
