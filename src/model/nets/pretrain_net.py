import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from src.model.nets import BaseNet


class PretrainMultitaskNet(BaseNet):
    """The Models Genesis network architecture for model pretraining in the multitask manner
    (image-to-image and domain classification).
    Ref:
        https://github.com/MrGiovanni/ModelsGenesis

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_domains (int): The number of the input domains.
    """

    def __init__(self, in_channels, out_channels, num_domains):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_domains = num_domains
        num_features = [32, 64, 128, 256, 512]

        self.in_block = _InBlock(in_channels, num_features[0], num_features[1])
        self.down_block1 = _DownBlock(num_features[1], num_features[2])
        self.down_block2 = _DownBlock(num_features[2], num_features[3])
        self.down_block3 = _DownBlock(num_features[3], num_features[4])

        self.up_block1 = _UpBlock(num_features[4] + num_features[3], num_features[3])
        self.up_block2 = _UpBlock(num_features[3] + num_features[2], num_features[2])
        self.up_block3 = _UpBlock(num_features[2] + num_features[1], num_features[1])
        self.out_block = _OutBlock(num_features[1], out_channels)

        self.out_linear = nn.Sequential()
        self.out_linear.add_module('linear1',
                                   nn.Linear(8 * 8 * 4 * 512, 512, bias=True))  # need to fix crop size as 64*64*32
        self.out_linear.add_module('norm', nn.BatchNorm1d(512, momentum=0.01, eps=1e-03))
        self.out_linear.add_module('relu', nn.ReLU(inplace=True))
        self.out_linear.add_module('linear2', nn.Linear(512, num_domains, bias=True))

    def forward(self, input):
        # Encoder
        features1 = self.in_block(input)
        features2 = self.down_block1(features1)
        features3 = self.down_block2(features2)
        features = self.down_block3(features3)

        # domain classifier
        domain_logits = self.out_linear(torch.flatten(features, start_dim=1)).unsqueeze(dim=-1)

        # Decoder
        features = self.up_block1(features, features3)
        features = self.up_block2(features, features2)
        features = self.up_block3(features, features1)
        image_logits = self.out_block(features)

        return image_logits, domain_logits


class PretrainDANet(BaseNet):
    """The Models Genesis network architecture for model pretraining with DANN-similar architecture.
    Ref:
        https://github.com/fungtion/DANN

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_domains (int): The number of the input domains.
    """

    def __init__(self, in_channels, out_channels, num_domains):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_domains = num_domains
        num_features = [32, 64, 128, 256, 512]

        self.in_block = _InBlock(in_channels, num_features[0], num_features[1])
        self.down_block1 = _DownBlock(num_features[1], num_features[2])
        self.down_block2 = _DownBlock(num_features[2], num_features[3])
        self.down_block3 = _DownBlock(num_features[3], num_features[4])

        self.up_block1 = _UpBlock(num_features[4] + num_features[3], num_features[3])
        self.up_block2 = _UpBlock(num_features[3] + num_features[2], num_features[2])
        self.up_block3 = _UpBlock(num_features[2] + num_features[1], num_features[1])
        self.out_block = _OutBlock(num_features[1], out_channels)

        self.out_linear = nn.Sequential()
        self.out_linear.add_module('linear1',
                                   nn.Linear(8 * 8 * 4 * 512, 512, bias=True))  # need to fix crop size as 64*64*32
        self.out_linear.add_module('norm', nn.BatchNorm1d(512, momentum=0.01, eps=1e-03))
        self.out_linear.add_module('relu', nn.ReLU(inplace=True))
        self.out_linear.add_module('linear2', nn.Linear(512, num_domains, bias=True))

    def forward(self, input):
        # Encoder
        features1 = self.in_block(input)
        features2 = self.down_block1(features1)
        features3 = self.down_block2(features2)
        features = self.down_block3(features3)

        # domain classifier
        reverse_features = _ReverseLayer.apply(features, 1.)
        domain_logits = self.out_linear(torch.flatten(reverse_features, start_dim=1)).unsqueeze(dim=-1)

        # Decoder
        features = self.up_block1(features, features3)
        features = self.up_block2(features, features2)
        features = self.up_block3(features, features1)
        image_logits = self.out_block(features)

        return image_logits, domain_logits


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


class _ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
