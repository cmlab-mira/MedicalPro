import re
import torch
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from torch._six import container_abcs, string_classes, int_classes

from src.data.datasets import BaseDataset
from src.data.transforms import Compose, ToTensor


np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


class VipcupSegDataset(BaseDataset):
    """The dataset of the Video and Image Processing Cup (VIPCUP) 2018
    for the lung cancer radiomics-tumor region segmentation task.

    Ref:
        https://signalprocessingsociety.org/get-involved/video-image-processing-cup

    Args:
        data_split_file_path (str): The data split file path.
        transforms (BoxList): The preprocessing techniques applied to the data.
        augments (BoxList): The augmentation techniques applied to the training data (default: None).
    """

    def __init__(self, data_split_file_path, transforms, augments=None, **kwargs):
        super().__init__(**kwargs)
        data_split_file = pd.read_csv(data_split_file_path)
        self.csv_name = Path(data_split_file_path).name
        patient_dirs = map(Path, data_split_file[data_split_file.type == self.type].path)
        self.data_paths = tuple(
            (patient_dir / f'{patient_dir.name}_img.nii.gz', 
            (
                patient_dir / f'{patient_dir.name}_img.nii.gz'
                if self.csv_name == 'testing.csv'
                else (
                    patient_dir / f'{patient_dir.name}_label.nii.gz'
                    if self.type == 'train'
                    else Path(
                        patient_dir.as_posix().replace('vipcup_resampled', 'vipcup')
                    ) / f'{patient_dir.name}_label.nii.gz'
                )
            ))
            for patient_dir in patient_dirs
        )
        self.transforms = Compose.compose(transforms)
        self.augments = Compose.compose(augments)
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        ct_path, gt_path = self.data_paths[index]
        nii_img = nib.load(ct_path.as_posix())
        ct = nii_img.get_fdata().astype(np.float32)[..., np.newaxis]
        gt = nib.load(gt_path.as_posix()).get_fdata().round().astype(np.int64)[..., np.newaxis]
        input_spacing = nii_img.header['pixdim'][1:4]

        ct, = self.transforms(ct)
        if self.type == 'train':
            ct, gt = self.augments(ct, gt)
        ct, gt = self.to_tensor(ct, gt)
        metadata = {'input': ct, 'target': gt}

        if self.type == 'test':
            metadata.update(affine=nii_img.affine,
                            header=nii_img.header,
                            name=ct_path.name.replace('_img', ''))
        return metadata

    def __len__(self):
        return len(self.data_paths)

    @classmethod
    def collate_fn(cls, batch):
        """Puts each data field into a tensor with outer dimension batch size
        Ref:
            https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        """
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return cls.collate_fn([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, nib.nifti1.Nifti1Header):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: cls.collate_fn([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(cls.collate_fn(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            transposed = zip(*batch)
            return [cls.collate_fn(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))
