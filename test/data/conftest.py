import pytest
from box import Box
from pathlib import Path
import numpy as np


@pytest.fixture
def config():
    cfg = Box.from_yaml(filename=Path("test/configs/test_config.yaml"))
    cfg.dataset.type = 'train'
    return cfg


@pytest.fixture
def dummy_input():
    def _generate(dim):
        if dim == 2:
            # B, H, W, C
            image = np.random.uniform(size=(32, 512, 512, 3))
            label = np.random.randint(0, 3, (32, 512, 512, 1))
        elif dim == 3:
            # B, H, W, D, C
            image = np.random.uniform(size=(32, 512, 512, 20, 3))
            label = np.random.randint(0, 3, (32, 512, 512, 20, 1))
        return image, label
    return _generate
