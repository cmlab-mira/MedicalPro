import pytest
import numpy as np
from box import Box
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.dataloader import Dataloader


class MyDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = np.random.uniform((1000, 512, 512))
        self.label = np.random.uniform((1000, 512, 512))

    def __getitem__(self, index):
        return {"input": self.data[index], "target": self.label[index]}

    def __len__(self):
        return self.data.shape[0]


class TestDataClass:

    @classmethod
    def setup_class(self):
        self.cfg = Box.from_yaml(filename=Path("test/configs/test_config.yaml"))
        self.cfg.dataset.type = 'train'

    def test_base_dataset(self):
        dataset = BaseDataset(**self.cfg.dataset)

    def test_base_dataset(self):
        dataset = MyDataset(**self.cfg.dataset)

    def test_data_loader(self):
        dataset = MyDataset(**self.cfg.dataset)
        dataloader = Dataloader(dataset, **self.cfg.dataloader)

        for batch in dataloader:
            assert batch['input'].shape == (self.cfg.dataloader.batch_size, 512, 512)
            assert batch['target'].shape == (self.cfg.dataloader.batch_size, 512, 512)
