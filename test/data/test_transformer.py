import torch
import pytest

from src.data.transformers import compose
from src.data.transformers import ToTensor
from src.data.transformers import Normalize
from src.data.transformers import RandomCrop


def test_composed_transformer(config, dummy_input):
    cfg = config
    transforms = compose(cfg.dataset.transforms)
    print(transforms)

    # B, H, W, C
    image, label = dummy_input(2)
    _image, _label = transforms(image[0], label[0], dtypes=[torch.float, torch.long])
    assert _image.dtype == torch.float
    assert _image.size() == (256, 256, image.shape[3])
    assert _label.dtype == torch.long
    assert _label.size() == (256, 256, label.shape[3])


def test_random_crop(dummy_input):
    # Test the 2D image: B, H, W, C
    image, label = dummy_input(2)
    transform = RandomCrop(size=(64, 64))
    _image, _label = transform(image[0], label[0])
    assert _image.shape == (64, 64, image.shape[3])
    assert _label.shape == (64, 64, label.shape[3])

    # Test the 3D image: B, H, W, D, C
    image, label = dummy_input(3)
    transform = RandomCrop(size=(64, 64, 8))
    _image, _label = transform(image[0], label[0])
    assert _image.shape == (64, 64, 8, image.shape[4])
    assert _label.shape == (64, 64, 8, label.shape[4])


def test_normalize(dummy_input):
    # Test the 2D image: B, H, W, C
    image, label = dummy_input(2)
    transform = Normalize(means=None, stds=None)
    _image, _label = transform(image[0], label[0], normalize_tags=[True, False])
    assert not (image[0] == _image).all()
    assert (label[0] == _label).all()

    # Test the 3D image: B, H, W, D, C
    image, label = dummy_input(3)
    transform = Normalize(means=None, stds=None)
    _image, _label = transform(image[0], label[0], normalize_tags=[True, False])
    assert not (image[0] == _image).all()
    assert (label[0] == _label).all()


def test_to_tensor(dummy_input):
    # Test the 2D image: B, H, W, C
    image, label = dummy_input(2)
    transform = ToTensor()
    _image, _label = transform(image, label, dtypes=[torch.float, torch.long])
    assert _image.dtype == torch.float
    assert _label.dtype == torch.long

    # Test the 3D image: B, H, W, D, C
    image, label = dummy_input(2)
    transform = ToTensor()
    _image, _label = transform(image, label, dtypes=[torch.float, torch.long])
    assert _image.dtype == torch.float
    assert _label.dtype == torch.long
