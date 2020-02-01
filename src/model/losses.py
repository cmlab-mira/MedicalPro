import torch
import torch.nn as nn

__all__ = [
    'CrossEntropyLossWrapper',
    'DiceLoss',
]


class CrossEntropyLossWrapper(nn.Module):
    """The cross-entropy loss wrapper.
    """

    def __init__(self, weight=None, **kwargs):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
            kwargs.update({'weight': weight})
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, *): The data target.

        Returns:
            loss (torch.Tensor) (0): The cross entropy loss.
        """
        return self.loss_fn(output, target)


class DiceLoss(nn.Module):
    """The Dice loss.
    Refs:
        https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895

    Args:
        smooth (float, optional): The smooth term (default: 1.0).
        square (bool, optional): Whether to use the square of cardinality (default: True).
        weight (list, optional): The weight given to each class (default: None).
    """

    def __init__(self, smooth=1.0, square=True, weight=None):
        super().__init__()
        self.smooth = smooth
        self.square = square
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
            self.register_buffer('weight', weight)

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.

        Returns:
            loss (torch.Tensor) (0): The dice loss.
        """
        # Get the one-hot encoding of the ground truth label.
        target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the dice loss.
        reduced_dims = list(range(2, output.dim()))  # (N, C, *) --> (N, C)
        intersection = (output * target).sum(reduced_dims)
        if self.square:
            cardinality = (output ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
        else:
            cardinality = output.sum(reduced_dims) + target.sum(reduced_dims)
        loss = 1 - (2 * intersection + self.smooth) / (cardinality + self.smooth)
        if getattr(self, 'weight', None) is not None:
            loss = (self.weight * loss).sum(dim=1).mean()
        else:
            loss = loss.mean()
        return loss
