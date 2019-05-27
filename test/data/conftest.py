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
    def _generate(image_size, label_size):
        image = np.random.uniform(size=image_size)
        label = np.random.randint(0, 3, size=label_size)
        return image, label
    return _generate
