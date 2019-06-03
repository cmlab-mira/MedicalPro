import torch.nn as nn
import numpy as np
from box import Box
from pathlib import Path

from src.model.nets.base_net import BaseNet


class MyNet(BaseNet):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(self.in_channels, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, self.out_channels, kernel_size=3)
    
    def forward(self, x):
        conv1 = self.conv1(x)
        output = self.conv2(conv1)
        return output


def test_base_net():
    """Test to build `BaseNet`.
    """
    net = BaseNet()


def test_my_net():
    """Test to build the derived network.
    """
    cfg = Box.from_yaml(filename=Path("test/configs/test_config.yaml"))
    net = MyNet(**cfg.net.kwargs)
