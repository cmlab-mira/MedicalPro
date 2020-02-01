import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.nets.base_net import BaseNet


class SegUNet3D(BaseNet):
    """
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_features (list): The list of the number of feature maps.
    """
    def __init__(self, in_channels, out_channels, num_features=[32, 64, 128, 256, 512], weight_path=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features

        self.in_block = _InBlock(in_channels, num_features[:2])
        self.down_block1 = _DownBlock(num_features[1], num_features[2])
        self.down_block2 = _DownBlock(num_features[2], num_features[3])
        self.down_block3 = _DownBlock(num_features[3], num_features[4])
        self.up_block1 = _UpBlock(num_features[4]+num_features[3], num_features[3])
        self.up_block2 = _UpBlock(num_features[3]+num_features[2], num_features[2])
        self.up_block3 = _UpBlock(num_features[2]+num_features[1], num_features[1])
        self.out_block = _OutBlock(num_features[1], out_channels)

        if weight_path is not None:
            self.load_state_dict(torch.load(weight_path))
        
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
    
    
class ClfUNet3D(BaseNet):
    """
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_features (list): The list of the number of feature maps.
    """
    def __init__(self, in_channels, out_channels, num_features=[32, 64, 128, 256, 512], weight_path=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features

        self.in_block = _InBlock(in_channels, num_features[:2])
        self.down_block1 = _DownBlock(num_features[1], num_features[2])
        self.down_block2 = _DownBlock(num_features[2], num_features[3])
        self.down_block3 = _DownBlock(num_features[3], num_features[4])
        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        self.dense_layer = nn.Sequential()
        self.dense_layer.add_module('fc1', nn.Linear(in_features=512, out_features=1024))
        self.dense_layer.add_module('relu1', nn.ReLU(inplace=True))
        self.dense_layer.add_module('fc2', nn.Linear(in_features=1024, out_features=out_channels))
        
        if weight_path is not None:
            self.load_state_dict(torch.load(weight_path))
        
    def forward(self, input):
        # Encoder
        features = self.in_block(input)
        features = self.down_block1(features)
        features = self.down_block2(features)
        features = self.down_block3(features)
        features = self.global_avgpool(features)
        features = torch.flatten(features, start_dim=1)
        output = self.dense_layer(features)
        
        return output


class _InBlock(nn.Sequential):
    def __init__(self, in_channels, num_features):
        super().__init__()
        self.add_module('conv1', nn.Conv3d(in_channels, num_features[0], kernel_size=3, padding=1))
        self.add_module('norm1', nn.BatchNorm3d(num_features[0], momentum=0.01, eps=1e-03))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(num_features[0], num_features[1], kernel_size=3, padding=1))
        self.add_module('norm2', nn.BatchNorm3d(num_features[1], momentum=0.01, eps=1e-03))
        self.add_module('relu2', nn.ReLU(inplace=True))


class _DownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('pool', nn.MaxPool3d(2))
        self.add_module('conv1', nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1))
        self.add_module('norm1', nn.BatchNorm3d(in_channels, momentum=0.01, eps=1e-03))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('norm2', nn.BatchNorm3d(out_channels, momentum=0.01, eps=1e-03))
        self.add_module('relu2', nn.ReLU(inplace=True))


class _UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.body = nn.Sequential()
        self.body.add_module('conv1', nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        self.body.add_module('norm1', nn.BatchNorm3d(out_channels, momentum=0.01, eps=1e-03))
        self.body.add_module('relu1', nn.ReLU(inplace=True))
        self.body.add_module('conv2', nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
        self.body.add_module('norm2', nn.BatchNorm3d(out_channels, momentum=0.01, eps=1e-03))
        self.body.add_module('relu2', nn.ReLU(inplace=True))

    def forward(self, input, features):
        input = self.upsample(input)
        d_diff = features.size(2) - input.size(2)
        h_diff = features.size(3) - input.size(3)
        w_diff = features.size(4) - input.size(4)
        input = F.pad(input, (w_diff // 2, w_diff - w_diff // 2,
                              h_diff // 2, h_diff - h_diff // 2,
                              d_diff // 2, d_diff - d_diff // 2,))
        output = self.body(torch.cat([input, features], dim=1))
        return output


class _OutBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=1))