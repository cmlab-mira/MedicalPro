import torch
from torchvision.transforms import Compose
import importlib


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
