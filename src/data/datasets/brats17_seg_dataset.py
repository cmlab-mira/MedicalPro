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


class Brats17SegDataset(BaseDataset):
    """The dataset of the Multimodal Brain Tumor Segmentation Challenge in MICCAI 2017 (BraTS17)
    for the segmentation task.

    Ref:
        https://www.med.upenn.edu/sbia/brats2017.html

    Args:
        data_split_file_path (str): The data split file path.
        transforms (BoxList): The preprocessing techniques applied to the data.
        augments (BoxList): The augmentation techniques applied to the training data (default: None).
    """

    def __init__(self, data_split_file_path, transforms, augments=None, **kwargs):
        super().__init__(**kwargs)
        data_split_file = pd.read_csv(data_split_file_path)
        patient_dirs = map(Path, data_split_file[data_split_file.type == self.type].path)
        self.data_paths = tuple(
            (
                patient_dir / f'{patient_dir.name}_t1.nii.gz',
                patient_dir / f'{patient_dir.name}_t1ce.nii.gz',
                patient_dir / f'{patient_dir.name}_t2.nii.gz',
                patient_dir / f'{patient_dir.name}_flair.nii.gz',
                patient_dir / f'{patient_dir.name}_seg.nii.gz',
            )
            for patient_dir in patient_dirs
        )
        self.transforms = Compose.compose(transforms)
        self.augments = Compose.compose(augments)
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        t1_path, t1ce_path, t2_path, flair_path, gt_path = self.data_paths[index]
        nii_img = nib.load(t1_path.as_posix())
        t1 = nii_img.get_fdata()
        t1ce = nib.load(t1ce_path.as_posix()).get_fdata()
        t2 = nib.load(t2_path.as_posix()).get_fdata()
        flair = nib.load(flair_path.as_posix()).get_fdata()
        mr = np.stack([t1, t1ce, t2, flair], axis=-1).astype(np.float32)
        gt = nib.load(gt_path.as_posix()).get_fdata().astype(np.int64)[..., np.newaxis]

        if self.type == 'train':
            transforms_kwargs = {
                'Clip': {
                    'transformed': (True, False)
                },
                'MinMaxScale': {
                    'transformed': (True, False),
                }
            }
            mr, gt = self.transforms(mr, gt, **transforms_kwargs)
            mr, gt = self.augments(mr, gt)
            mr, gt = self.to_tensor(mr, gt)
        else:
            mr, = self.transforms(mr, **transforms_kwargs)
            mr, gt = self.to_tensor(mr, gt)
        metadata = {'input': mr, 'target': gt}

        if self.type == 'test':
            metadata.update(affine=nii_img.affine,
                            header=nii_img.header,
                            name=t1_path.parent)
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
