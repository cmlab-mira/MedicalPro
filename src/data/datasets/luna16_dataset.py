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
    def __init__(self, data_dir, transforms, positive_ratio=None, train_fold=None, 
                 valid_fold=None, test_fold=None, augments=None, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)
        self.positive_ratio = positive_ratio
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
        if positive_ratio is not None:
            negative_data_paths = np.random.choice(negative_data_paths, int(len(positive_data_paths) / positive_ratio), replace=False)
        self.data_paths = np.concatenate((positive_data_paths, negative_data_paths), axis=-1)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        path = self.data_paths[index]
        img = np.load(path)[..., None]
        label = 0 if path.parts[-3] == 'negative' else 1
        
        """
        sid = path.name.replace('.mhd', '')
        
        if random.random() < self.positive_prob:
            # Positive sampling
            df = pd.read_csv(self.positive_csv)
            label = 1
        else:
            # Negative sampling
            df = pd.read_csv(self.negative_csv)
            label = 0
            
        # Get the candidate information
        info = df[df.seriesuid == sid]
        try:
            word_coord = np.array(info.values[0][1:4])
        except:
            # There is no positive sample belonging to that patient. 
            # Instead, use the negative sample
            df = pd.read_csv(self.negative_csv)
            label = 0
            info = df[df.seriesuid == sid]
            word_coord = np.array(info.values[0][1:4])
            
        # Read the CT image
        itk_img = sitk.ReadImage(str(path))
        origin = np.array(list(itk_img.GetOrigin()))
        spacing = np.array(list(itk_img.GetSpacing()))
        img_array = sitk.GetArrayFromImage(itk_img)
        img_array = img_array.transpose(2, 1, 0)
        
        # Execute the preprocessing according to the ModelGenesis guildlines
        # - all the intensity values be clipped on the min (-1000) and max (+1000)
        img_array = img_array.clip(-1000, 1000)
        # - scale between 0 and 1
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        
        voxel_coord = self._world_to_voxel_coord(word_coord, origin, spacing)
        cropped_img = self._crop(img_array, voxel_coord)[..., None] # H, W, D, C
        """
        
        if self.type == 'Train' and self.augments is not None:
            img = self.augments(img)
        data = self.transforms(img, dtypes=[torch.float]).permute(3, 2, 0, 1).contiguous()
        label = self.transforms(np.array([label]), dtypes=[torch.long])
        return {'data': data, 'label': label}
    
    def _world_to_voxel_coord(self, world_coord, origin, spacing):
        stretched_voxel_coord = np.absolute(world_coord - origin)
        voxel_coord = (stretched_voxel_coord / spacing)
        voxel_coord = [np.round(coord).astype(np.int) for coord in voxel_coord]
        return voxel_coord
    
    def _crop(self, img, coord):
        x_start, x_end = coord[0]-self.crop_size[0]//2, coord[0]+self.crop_size[0]//2
        y_start, y_end = coord[1]-self.crop_size[1]//2, coord[1]+self.crop_size[1]//2
        z_start, z_end = coord[2]-self.crop_size[2]//2, coord[2]+self.crop_size[2]//2
        
        if x_start < 0:
            x_start, x_end = 0, self.crop_size[0]
        if x_end >= img.shape[0]:
            x_start, x_end = img.shape[0]-self.crop_size[0], img.shape[0]
        
        if y_start < 0:
            y_start, y_end = 0, self.crop_size[1]
        if y_end >= img.shape[1]:
            y_start, y_end = img.shape[1]-self.crop_size[1], img.shape[1]
        
        if z_start < 0:
            z_start, z_end = 0, self.crop_size[2]
        if z_end >= img.shape[2]:
            z_start, z_end = img.shape[2]-self.crop_size[2], img.shape[2]
            
        cropped_img = img[x_start:x_end, y_start:y_end, z_start:z_end]
        return cropped_img