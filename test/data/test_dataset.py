import pytest
import numpy as np
from box import Box
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.dataloader import Dataloader


class MyDataset(BaseDataset):

    def __init__(self, dummy_input, **kwargs):
        super().__init__(**kwargs)
        self.data, self.label = dummy_input;

    def __getitem__(self, index):
        return {"input": self.data[index], "target": self.label[index]}

    def __len__(self):
        return self.data.shape[0]


def test_base_dataset(config):
    cfg = config
    dataset = BaseDataset(**cfg.dataset)


def test_base_dataset(config, dummy_input):
    cfg = config
    dataset = MyDataset(dummy_input(dim=2), **cfg.dataset)


def test_data_loader(config, dummy_input):
    cfg = config
    image, label = dummy_input(dim=2)
    dataset = MyDataset(dummy_input(dim=2), **cfg.dataset)
    dataloader = Dataloader(dataset, **cfg.dataloader)

    for batch in dataloader:
        assert batch['input'].shape == (cfg.dataloader.batch_size, *image.shape[1:])
        assert batch['target'].shape == (cfg.dataloader.batch_size, *label.shape[1:])
