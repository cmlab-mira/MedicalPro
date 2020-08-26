import torch
import torch.nn.functional as F
import numpy as np
import copy

from src.runner.trainers import BaseTrainer


class Brats17SegTrainer(BaseTrainer):
    """The BraTS trainer for the segmentation task.
    """

    def __init__(self, gamma_threshold=None, half_bn_grad=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma_threshold = gamma_threshold
        self.half_bn_grad = half_bn_grad
        if gamma_threshold is not None:
            self._init_random_indices()

    def _train_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target'].to(self.device)
        output = self.net(input)
        cross_entropy_loss = self.loss_fns.cross_entropy_loss(output, target.squeeze(dim=1))
        dice_loss = self.loss_fns.dice_loss(output, target)

        if getattr(self.loss_fns, "bn_gamma_loss", None) is not None:
            bn_gamma_loss = self.loss_fns.bn_gamma_loss(self.net)
            loss = (
                self.loss_weights.cross_entropy_loss * cross_entropy_loss +
                self.loss_weights.dice_loss * dice_loss +
                self.loss_weights.bn_gamma_loss * bn_gamma_loss
            )
            return {
                'loss': loss,
                'losses': {
                    'CrossEntropyLoss': cross_entropy_loss,
                    'DiceLoss': dice_loss,
                    'BatchNormGammaLoss': bn_gamma_loss,
                }
            }
        else:
            loss = (
                self.loss_weights.cross_entropy_loss * cross_entropy_loss +
                self.loss_weights.dice_loss * dice_loss
            )
            return {
                'loss': loss,
                'losses': {
                    'CrossEntropyLoss': cross_entropy_loss,
                    'DiceLoss': dice_loss,
                }
            }

    def _valid_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target'].to(self.device)
        output = self.net(input)
        output = F.interpolate(output,
                               size=target.size()[2:],
                               mode='trilinear',
                               align_corners=False)
        
        cross_entropy_loss = self.loss_fns.cross_entropy_loss(output, target.squeeze(dim=1))
        dice_loss = self.loss_fns.dice_loss(output, target)
        loss = (self.loss_weights.cross_entropy_loss * cross_entropy_loss +
                self.loss_weights.dice_loss * dice_loss)

        # Enhancing tumor: label 3
        output_prob = F.softmax(output, dim=1)
        output_enhancing_tumor = self.merge_prob(
            output_prob,
            background_indices=[0, 1, 2],
            foreground_indices=[3]
        )
        dice_enhancing_tumor = self.metric_fns.dice(
            output_enhancing_tumor,
            (target == 3).to(torch.long)
        )[1]

        return {
            'loss': loss,
            'losses': {
                'CrossEntropyLoss': cross_entropy_loss,
                'DiceLoss': dice_loss,
            },
            'metrics': {
                'DiceEnhancingTumor': dice_enhancing_tumor
            }
        }

    @staticmethod
    def merge_prob(prob, background_indices, foreground_indices):
        background_indices = torch.as_tensor(background_indices, dtype=torch.long, device=prob.device)
        foreground_indices = torch.as_tensor(foreground_indices, dtype=torch.long, device=prob.device)
        background_prob = prob.index_select(dim=1, index=background_indices).sum(dim=1, keepdim=True)
        foreground_prob = prob.index_select(dim=1, index=foreground_indices).sum(dim=1, keepdim=True)
        merged_prob = torch.cat([background_prob, foreground_prob], dim=1)
        return merged_prob

    def _init_random_indices(self):
        training_indices = {}
        self.frozen_indices = {}
        for key in self.net.norm_trainable_only_state_dict.keys():
            if ('norm' in key) and ('weight' in key):
                _key = key.replace('.weight', '')
                self.frozen_indices[_key] = torch.where(self.net.norm_trainable_only_state_dict[key].abs() > self.gamma_threshold)[0]
                self.frozen_indices[_key.replace('norm', 'conv')] = self.frozen_indices[_key]
                training_indices[_key] = torch.where(self.net.norm_trainable_only_state_dict[key].abs() < self.gamma_threshold)[0]
                training_indices[_key.replace('norm', 'conv')] = training_indices[_key]

        # Reset the parameters needed to be trained to the randomly initialized state
        for key in self.net.state_dict().keys():
            _key = '.'.join(key.split('.')[:-1])
            if _key in training_indices and 'num_batches_tracked' not in key:
                self.net.state_dict()[key][training_indices[_key]] = self.net.random_init_state_dict[key][training_indices[_key]].to(self.device)

    def _modify_grad(self):
        if self.gamma_threshold is not None:
            for name, param in self.net.named_parameters():
                key = '.'.join(name.split('.')[:-1])
                if key in self.frozen_indices:
                    param.grad[self.frozen_indices[key]] *= 0
        if self.half_bn_grad is True:
            for block_name, block in self.net.named_children():
                for module_name, module in block.named_modules():
                    if 'BatchNorm' in module.__class__.__name__:
                        for name, param in module.named_parameters():
                            if name == 'weight':
                                param.grad *= 0.5
