import torch
from torchvision.transforms import Compose
import importlib
import numpy as np


def compose(transforms):
    """Compose several transformers together.
    Args:
        transforms (Box): The preprocessing and augmentation techniques applied to the data.

    Returns:
        transforms (list of ``Transformer`` objects): The list of transformers.
    """
    _transforms = []
    for transform in transforms:
        if transform.do:
            cls_name = ''.join([str_.capitalize() for str_ in tranform.split('_'))
            cls = getattr(importlib.import_module('src.data.transformers'), cls_name)
            _transforms.append(cls(**transform.kwargs))

    # Append the default transformer ``ToTensor``
    _transforms.append(ToTensor())

    transforms = Compose(_transforms)
    return transforms


class BaseTransformer:
    """The base class for all transformers.
    """
    def __call__(self, *imgs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class ToTensor:
    """Convert a tuple of ``numpy.ndarray`` to a tuple of ``torch.Tensor``.
    """
    def __call__(self, *imgs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be converted to tensor.

        Returns:
            imgs (tuple of torch.Tensor): The converted images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
        
        # (H, W, C) -> (C, H, W); (H, W, D, C) -> (C, D, H, W)
        if all(img.ndim == 3 for img in imgs):
            imgs = tuple(img.float().permute(2, 0, 1).contiguous() for img in map(torch.from_numpy, imgs))
        elif all(img.ndim == 4 for img in imgs):
            imgs = tuple(img.float().permute(3, 2, 0, 1).contiguous() for img in map(torch.from_numpy, imgs))
        else:
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")
        return imgs
