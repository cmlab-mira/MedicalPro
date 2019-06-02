import torch
import pytest
import numpy as np

from src.data.transformers import compose
from src.data.transformers import ToTensor
from src.data.transformers import Normalize
from src.data.transformers import RandomCrop
from src.data.transformers import Resize


def test_composed_transformer(config, dummy_input):
    """Test to compose multiple augmentations
    including RandomCrop, Normalize, ToTensor.
    """
    cfg = config
    transforms = compose(cfg.dataset.transforms)
    print(transforms)

    # H, W, C
    image, label = dummy_input(image_size=(512, 512, 3),
                               label_size=(512, 512, 1))
    _image, _label = transforms(image, label, dtypes=[torch.float, torch.long])
    assert _image.dtype == torch.float
    assert _image.size() == (256, 256, image.shape[2])
    assert _label.dtype == torch.long
    assert _label.size() == (256, 256, label.shape[2])


def test_random_crop(dummy_input):
    """Test random cropping the 2D and 3D images.
    """
    # Test the 2D image: H, W, C
    image, label = dummy_input(image_size=(512, 512, 3),
                               label_size=(512, 512, 1))
    transform = RandomCrop(size=(64, 64))
    _image, _label = transform(image, label)
    assert _image.shape == (64, 64, image.shape[2])
    assert _label.shape == (64, 64, label.shape[2])

    # Test the 3D image: H, W, D, C
    image, label = dummy_input(image_size=(512, 512, 20, 3),
                               label_size=(512, 512, 20, 1))
    transform = RandomCrop(size=(64, 64, 8))
    _image, _label = transform(image, label)
    assert _image.shape == (64, 64, 8, image.shape[3])
    assert _label.shape == (64, 64, 8, label.shape[3])


def test_normalize(dummy_input):
    """Test to normalize the 2D and 3D images with specific tags
    to indicate whether to perform normalization to the object.
    """
    # Test the 2D image: H, W, C
    image, label = dummy_input(image_size=(512, 512, 3),
                               label_size=(512, 512, 1))
    transform = Normalize(means=None, stds=None)
    _image, _label = transform(image, label, normalize_tags=[True, False])
    assert not (image == _image).all()
    assert (label == _label).all()

    # Test the 3D image: H, W, D, C
    image, label = dummy_input(image_size=(512, 512, 20, 3),
                               label_size=(512, 512, 20, 1))
    transform = Normalize(means=None, stds=None)
    _image, _label = transform(image, label, normalize_tags=[True, False])
    assert not (image == _image).all()
    assert (label == _label).all()
    assert np.abs(np.mean(_image)-0) < 1e-8
    assert np.abs(np.std(_image)-1) < 1e-8


def test_resize(dummy_input):
    """Test to normalize the 2D and 3D images with specific tags
    to indicate whether to perform normalization to the object.
    """
    # Test the 2D image: H, W, C
    image, label = dummy_input(image_size=(512, 512, 3),
                               label_size=(512, 512, 1))
    transform = Resize(size=(64, 64))
    _image, _label = transform(image, label, resize_orders=[3, 0])
    assert _image.shape == (64, 64, 3)
    assert _image.dtype == image.dtype
    assert _label.shape == (64, 64, 1)
    assert _label.dtype == label.dtype

    # Test the 3D image: H, W, D, C
    image, label = dummy_input(image_size=(512, 512, 20, 3),
                               label_size=(512, 512, 20, 1))
    transform = Resize(size=(64, 64, 10))
    _image, _label = transform(image, label, resize_orders=[3, 0])
    assert _image.shape == (64, 64, 10, 3)
    assert _image.dtype == image.dtype
    assert _label.shape == (64, 64, 10, 1)
    assert _label.dtype == label.dtype


def test_to_tensor(dummy_input):
    """Test to convert the input numpy array to torch tensor.
    `dtypes` can be used to assign the output tensor types.
    """
    # Test the 2D image: B, H, W, C
    image, label = dummy_input(image_size=(512, 512, 3),
                               label_size=(512, 512, 1))
    transform = ToTensor()
    _image, _label = transform(image, label, dtypes=[torch.float, torch.long])
    assert _image.dtype == torch.float
    assert _label.dtype == torch.long

    # Test the 3D image: B, H, W, D, C
    image, label = dummy_input(image_size=(512, 512, 20, 3),
                               label_size=(512, 512, 20, 1))
    transform = ToTensor()
    _image, _label = transform(image, label, dtypes=[torch.float, torch.long])
    assert _image.dtype == torch.float
    assert _label.dtype == torch.long
