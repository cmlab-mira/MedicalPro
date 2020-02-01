import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.nets import BaseNet


class UNet3D(BaseNet):
    """The 3D U-Net.
    Ref:
        https://arxiv.org/pdf/1606.06650.pdf

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_features (list): The list of the number of feature maps.
    """

    def __init__(self, in_channels, out_channels, num_features=[32, 64, 128, 256, 512]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features

        self.in_block = _InBlock(in_channels, num_features[0], num_features[1])
        self.down_block1 = _DownBlock(num_features[1], num_features[2])
        self.down_block2 = _DownBlock(num_features[2], num_features[3])
        self.down_block3 = _DownBlock(num_features[3], num_features[4])
        self.up_block1 = _UpBlock(num_features[4], num_features[3])
        self.up_block2 = _UpBlock(num_features[3], num_features[2])
        self.up_block3 = _UpBlock(num_features[2], num_features[1])
        self.out_block = _OutBlock(num_features[1], out_channels)

    def forward(self, input):
        # Encoder
        features1 = self.in_block(input)
        features2 = self.down_block1(features1)
        features3 = self.down_block2(features2)
        features = self.down_block3(features3)

        # Decoder
        features = self.up_block1(features, features3)
        features = self.up_block2(features, features2)
        features = self.up_block3(features, features1)
        output = self.out_block(features)
        return output


class _InBlock(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.add_module('conv1', nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1))
        self.add_module('norm1', nn.BatchNorm3d(mid_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('norm2', nn.BatchNorm3d(out_channels))
        self.add_module('relu2', nn.ReLU(inplace=True))


class _DownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('pool', nn.MaxPool3d(2))
        self.add_module('conv1', nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1))
        self.add_module('norm1', nn.BatchNorm3d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('norm2', nn.BatchNorm3d(out_channels))
        self.add_module('relu2', nn.ReLU(inplace=True))


class _UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(nn.Conv3d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(out_channels),
                                  nn.ReLU(inplace=True))

    def forward(self, input, features):
        input = self.deconv(input)
        d_diff = features.size(2) - input.size(2)
        h_diff = features.size(3) - input.size(3)
        w_diff = features.size(4) - input.size(4)
        input = F.pad(input, (w_diff // 2, w_diff - w_diff // 2,
                              h_diff // 2, h_diff - h_diff // 2,
                              d_diff // 2, d_diff - d_diff // 2,))
        output = self.conv(torch.cat([input, features], dim=1))
        return output


class _OutBlock(nn.Conv3d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1)
