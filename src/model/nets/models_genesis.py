import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.model.nets import BaseNet


class ModelsGenesisSegNet(BaseNet):
    """The Models Genesis network architecture for segmentation task (similar to the 3D U-Net).
    Ref:
        https://github.com/MrGiovanni/ModelsGenesis

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        weight_path (str, optional): The pre-trained weight path (default: None).
        loaded_modules (sequence, optional): Specify which modules should be loaded (default: None, all modules).
            Note that this argument is only valid when weight_path is not None.
        frozen_modules (sequence, optional): Specify which modules should be frozen (default: None).
    """

    def __init__(self, in_channels, out_channels, weight_path=None, loaded_modules=None, frozen_modules=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_features = [32, 64, 128, 256, 512]

        self.in_block = _InBlock(in_channels, num_features[0], num_features[1])
        self.down_block1 = _DownBlock(num_features[1], num_features[2])
        self.down_block2 = _DownBlock(num_features[2], num_features[3])
        self.down_block3 = _DownBlock(num_features[3], num_features[4])
        self.up_block1 = _UpBlock(num_features[4] + num_features[3], num_features[3])
        self.up_block2 = _UpBlock(num_features[3] + num_features[2], num_features[2])
        self.up_block3 = _UpBlock(num_features[2] + num_features[1], num_features[1])
        self.out_block = _OutBlock(num_features[1], out_channels)

        if weight_path is not None:
            state_dict = self.state_dict()
            if Path(weight_path).name == 'models_genesis.pth':
                pretrained_state_dict = torch.load(weight_path, map_location='cpu')
            else:  # For our saved model checkpoints.
                pretrained_state_dict = torch.load(weight_path, map_location='cpu')['net']
                pretrained_state_dict.pop('out_block.weight')
                pretrained_state_dict.pop('out_block.bias')
            if loaded_modules is not None:
                assert len(loaded_modules) > 0
                loaded_modules = set(loaded_modules)
                if 'encoder' in loaded_modules:
                    loaded_modules.remove('encoder')
                    loaded_modules.update({'in_block', 'down_block1', 'down_block2', 'down_block3'})
                if 'decoder' in loaded_modules:
                    loaded_modules.remove('decoder')
                    loaded_modules.update({'up_block1', 'up_block2', 'up_block3'})
                all_modules = {'in_block', 'down_block1', 'down_block2', 'down_block3',
                               'up_block1', 'up_block2', 'up_block3'}
                assert loaded_modules.issubset(all_modules)
                for key in list(pretrained_state_dict.keys()):
                    if not any(loaded_module in key for loaded_module in loaded_modules):
                        pretrained_state_dict.pop(key)
            state_dict.update(pretrained_state_dict)
            self.load_state_dict(state_dict)

        if frozen_modules is not None:
            assert 'out_block' not in frozen_modules
            frozen_modules = set(frozen_modules)
            if 'encoder' in frozen_modules:
                frozen_modules.remove('encoder')
                frozen_modules.update(['in_block', 'down_block1', 'down_block2', 'down_block3'])
            if 'decoder' in frozen_modules:
                frozen_modules.remove('decoder')
                frozen_modules.update(['up_block1', 'up_block2', 'up_block3'])
            for param in itertools.chain.from_iterable(
                getattr(self, frozen_module).parameters()
                for frozen_module in frozen_modules
            ):
                param.requires_grad = False

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


class ModelsGenesisClfNet(BaseNet):
    """The Models Genesis network architecture for classification task.
    Ref:
        https://github.com/MrGiovanni/ModelsGenesis

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        weight_path (str, optional): The pre-trained weight path (default: None).
        loaded_modules (sequence, optional): Specify which modules should be loaded (default: None, encoder).
            Note that this argument is only valid when weight_path is not None.
        frozen_modules (sequence, optional): Specify which modules should be frozen (default: None).
    """

    def __init__(self, in_channels, out_channels, weight_path=None, loaded_modules=None, frozen_modules=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_features = [32, 64, 128, 256, 512, 1024]

        self.in_block = _InBlock(in_channels, num_features[0], num_features[1])
        self.down_block1 = _DownBlock(num_features[1], num_features[2])
        self.down_block2 = _DownBlock(num_features[2], num_features[3])
        self.down_block3 = _DownBlock(num_features[3], num_features[4])
        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential()
        self.classifier.add_module('fc1', nn.Linear(in_features=num_features[4],
                                                    out_features=num_features[5]))
        self.classifier.add_module('relu1', nn.ReLU(inplace=True))
        self.classifier.add_module('fc2', nn.Linear(in_features=num_features[5],
                                                    out_features=out_channels))

        if weight_path is not None:
            state_dict = self.state_dict()
            if Path(weight_path).name == 'models_genesis.pth':
                pretrained_state_dict = torch.load(weight_path, map_location='cpu')
            else:  # For our saved model checkpoints.
                pretrained_state_dict = torch.load(weight_path, map_location='cpu')['net']
            pretrained_state_dict = {key: value for key, value in pretrained_state_dict.items()
                                     if key in state_dict.keys()}
            if loaded_modules is not None:
                assert len(loaded_modules) > 0
                loaded_modules = set(loaded_modules)
                if 'encoder' in loaded_modules:
                    loaded_modules.remove('encoder')
                    loaded_modules.update({'in_block', 'down_block1', 'down_block2', 'down_block3'})
                all_modules = {'in_block', 'down_block1', 'down_block2', 'down_block3'}
                assert loaded_modules.issubset(all_modules)
                for key in list(pretrained_state_dict.keys()):
                    if not any(loaded_module in key for loaded_module in loaded_modules):
                        pretrained_state_dict.pop(key)
            state_dict.update(pretrained_state_dict)
            self.load_state_dict(state_dict)

        if frozen_modules is not None:
            assert 'classifier' not in frozen_modules
            frozen_modules = set(frozen_modules)
            if 'encoder' in frozen_modules:
                frozen_modules.remove('encoder')
                frozen_modules.update(['in_block', 'down_block1', 'down_block2', 'down_block3'])
            for param in itertools.chain.from_iterable(
                getattr(self, frozen_module).parameters()
                for frozen_module in frozen_modules
            ):
                param.requires_grad = False

    def forward(self, input):
        # Encoder
        features = self.in_block(input)
        features = self.down_block1(features)
        features = self.down_block2(features)
        features = self.down_block3(features)
        features = self.global_avgpool(features)

        # Classifier
        features = torch.flatten(features, start_dim=1)
        output = self.classifier(features)
        return output


class _InBlock(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.add_module('conv1', nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1))
        self.add_module('norm1', nn.BatchNorm3d(mid_channels, momentum=0.01, eps=1e-03))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('norm2', nn.BatchNorm3d(out_channels, momentum=0.01, eps=1e-03))
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


class _OutBlock(nn.Conv3d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1)
