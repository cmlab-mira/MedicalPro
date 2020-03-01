import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
import SimpleITK as sitk

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class Luna16Dataset(BaseDataset):
    """The dataset of false positive reduction track in LUNA16 challenge.
    Args:
        data_dir (str): The data folder path.
        fold (list of int): The indices of the files used in the training/validation/testing.
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
    """
    def __init__(self, data_dir, transforms, positive_ratio=None, img_size=[16, 16, 16],
                 train_fold=None, valid_fold=None, test_fold=None, augments=None, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)
        self.positive_ratio = positive_ratio
        self.img_size = img_size
        self.transforms = compose(transforms)
        self.augments = compose(augments)
        
        if self.type == 'train':
            self.fold = train_fold
        elif self.type == 'valid':
            self.fold = valid_fold
        elif self.type == 'test':
            self.fold = test_fold
        
        positive_data_paths, negative_data_paths = [], []
        for fold in self.fold:
            positive_subset_path = self.data_dir / 'positive' / f"subset{fold}"
            positive_file_list = list(sorted(positive_subset_path.glob("./*.npy")))
            positive_data_paths.extend(positive_file_list)
            
            negative_subset_path = self.data_dir / 'negative' / f"subset{fold}"
            negative_file_list = list(sorted(negative_subset_path.glob("./*.npy")))
            negative_data_paths.extend(negative_file_list)
            
        positive_data_paths = np.array(positive_data_paths)
        negative_data_paths = np.array(negative_data_paths)
        if (positive_ratio is not None) and (self.type == 'train'):
            negative_data_paths = np.random.choice(negative_data_paths, int(len(positive_data_paths) / positive_ratio), replace=False)
        self.data_paths = np.concatenate((positive_data_paths, negative_data_paths), axis=-1)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        path = self.data_paths[index]
        img = np.load(path)[..., None]
        label = 0 if path.parts[-3] == 'negative' else 1
        
        if (self.type == 'train') and (self.augments is not None):
            img = self.augments(img)
        
        if not np.array_equal(img.shape[:-1], self.img_size):
            img = self._crop(img, self.img_size)
            
        data = self.transforms(img, dtypes=[torch.float]).permute(3, 2, 0, 1).contiguous()
        label = self.transforms(np.array([label]), dtypes=[torch.long])
        if self.type == 'test':
            cid = int(path.stem.split('_')[1])
            return {'data': data, 'label': label, 'cid': cid}
        else:
            return {'data': data, 'label': label}
    
    def _world_to_voxel_coord(self, world_coord, origin, spacing):
        stretched_voxel_coord = np.absolute(world_coord - origin)
        voxel_coord = (stretched_voxel_coord / spacing)
        voxel_coord = [np.round(coord).astype(np.int) for coord in voxel_coord]
        return voxel_coord
    
    def _crop(self, img, size):
        if img.ndim == 3:
            H, W, _ = img.shape
            x_start, x_end = H // 2 - size[0] // 2, H // 2 + size[0] // 2
            y_start, y_end = W // 2 - size[1] // 2, W // 2 + size[1] // 2
            cropped_img = img[x_start:x_end, y_start:y_end]
        elif img.ndim == 4:
            H, W, D, _ = img.shape
            x_start, x_end = H // 2 - size[0] // 2, H // 2 + size[0] // 2
            y_start, y_end = W // 2 - size[1] // 2, W // 2 + size[1] // 2
            z_start, z_end = D // 2 - size[2] // 2, D // 2 + size[2] // 2
            cropped_img = img[x_start:x_end, y_start:y_end, z_start:z_end]
        return cropped_img