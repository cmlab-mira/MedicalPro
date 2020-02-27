import torch
import torch.nn as nn

__all__ = [
    'CrossEntropyLossWrapper',
    'DiceLoss',
    'TverskyLoss',
]


class CrossEntropyLossWrapper(nn.Module):
    """The cross-entropy loss wrapper.
    Args:
        weight (sequence, optional): The weight given to each class (default: None).
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
            output (torch.Tensor) (N, C, *): The output logits.
            target (torch.LongTensor) (N, *): The target where each value is between 0 and C-1.

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
        weight (sequence, optional): The weight given to each class (default: None).
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
            output (torch.Tensor) (N, C, *): The output probability.
            target (torch.LongTensor) (N, 1, *): The target where each value is between 0 and C-1.

        Returns:
            loss (torch.Tensor) (0): The dice loss.
        """
        # Get the one-hot encoding of the ground truth label.
        target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the dice loss.
        reduced_dims = tuple(range(2, output.dim()))  # (N, C, *) --> (N, C)
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


class TverskyLoss(nn.Module):
    """The Tversky loss.
    Refs:
        https://arxiv.org/abs/1706.05721

    Args:
        alpha (float): The magnitude of penalties for the False Positives (FPs).
        beta (float): The magnitude of penalties for the False Negatives (FNs).
        smooth (float, optional): The smooth term (default: 1.0).
        weight (sequence, optional): The weight given to each class (default: None).
    """

    def __init__(self, alpha, beta, smooth=1.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
            self.register_buffer('weight', weight)

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The output probability.
            target (torch.LongTensor) (N, 1, *): The target where each value is between 0 and C-1.

        Returns:
            loss (torch.Tensor) (0): The tversky loss.
        """
        # Get the one-hot encoding of the ground truth label.
        target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the tversky loss.
        reduced_dims = tuple(range(2, output.dim()))  # (N, C, *) --> (N, C)
        intersection = (output * target).sum(reduced_dims)
        fps = (output * (1 - target)).sum(reduced_dims)
        fns = ((1 - output) * target).sum(reduced_dims)
        loss = 1 - (intersection + self.smooth) / (intersection + self.alpha * fps + self.beta * fns + self.smooth)
        if getattr(self, 'weight', None) is not None:
            loss = (self.weight * loss).sum(dim=1).mean()
        else:
            loss = loss.mean()
        return loss
