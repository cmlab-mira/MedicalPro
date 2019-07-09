import pytest
import numpy as np
from box import Box
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.dataloader import Dataloader
from src.data.transforms import compose


class MyDataset(BaseDataset):
    """The derived class of BaseDataset.
    """
    def __init__(self, image, label, transforms, **kwargs):
        super().__init__(**kwargs)
        self.image, self.label = image, label
        self.transforms = compose(transforms)

    def __getitem__(self, index):
        return {"input": self.image[index], "target": self.label[index]}

    def __len__(self):
        return self.image.shape[0]


def test_base_dataset(config):
    """Test to create `baseDataset`.
    """
    cfg = config
    cfg.dataset.pop('transforms')
    dataset = BaseDataset(**cfg.dataset)


def test_my_dataset(config, dummy_input):
    """Test to create the derived dataset.
    """
    cfg = config
    image, label = dummy_input(image_size=(10, 32, 32, 3),
                               label_size=(10, 32, 32, 1))
    dataset = MyDataset(image, label, **cfg.dataset)


def test_data_loader(config, dummy_input):
    """Test to create the dataloader and yield a batch of data.
    """
    cfg = config
    image, label = dummy_input(image_size=(10, 32, 32, 3),
                               label_size=(10, 32, 32, 1))
    dataset = MyDataset(image, label, **cfg.dataset)
    dataloader = Dataloader(dataset, **cfg.dataloader)

    for batch in dataloader:
        assert batch['input'].shape == (cfg.dataloader.batch_size, *image.shape[1:])
        assert batch['target'].shape == (cfg.dataloader.batch_size, *label.shape[1:])
