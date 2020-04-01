import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'BCELossWrapper',
    'CrossEntropyLossWrapper',
    'DiceLoss',
    'TverskyLoss',
]


class BCELossWrapper(nn.Module):
    """The binary cross-entropy loss wrapper which combines torch.nn.BCEWithLogitsLoss (with logits)
    and torch.nn.BCELoss (with probability).

    Ref: 
        https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss
        https://pytorch.org/docs/stable/nn.html#bceloss

    Args:
        with_logits (bool, optional): Specify the output is logits or probability (default: True).
        weight (sequence, optional): The same argument in torch.nn.BCEWithLogitsLoss and torch.nn.BCELoss
            but its type is sequence for the configuration purpose (default: None).
        pos_weight (sequence, optional): The same argument in torch.nn.BCEWithLogitsLoss
            but its type is sequence for the configuration purpose (default: None).
    """

    def __init__(self, with_logits=True, weight=None, pos_weight=None, **kwargs):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
            kwargs.update(weight=weight)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float)
            kwargs.update(pos_weight=pos_weight)
        self.loss_fn = (nn.BCEWithLogitsLoss if with_logits else nn.BCELoss)(**kwargs)

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, *): The output logits or probability.
            target (torch.LongTensor) (N, *): The target where each value is 0 or 1.

        Returns:
            loss (torch.Tensor) (0): The binary cross entropy loss.
        """
        return self.loss_fn(output, target)


class CrossEntropyLossWrapper(nn.Module):
    """The cross-entropy loss wrapper which combines torch.nn.CrossEntropyLoss (with logits)
    and torch.nn.NLLLoss (with probability).

    Ref: 
        https://pytorch.org/docs/stable/nn.html#crossentropyloss
        https://pytorch.org/docs/stable/nn.html#nllloss

    Args:
        with_logits (bool, optional): Specify the output is logits or probability (default: True).
        weight (sequence, optional): The same argument in torch.nn.CrossEntropyLoss and torch.nn.NLLLoss
            but its type is sequence for the configuration purpose (default: None).
    """

    def __init__(self, with_logits=True, weight=None, **kwargs):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
            kwargs.update(weight=weight)
        self.loss_fn = (nn.CrossEntropyLoss if with_logits else nn.NLLLoss)(**kwargs)

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The output logits or probability.
            target (torch.LongTensor) (N, *): The target where each value is between 0 and C-1.

        Returns:
            loss (torch.Tensor) (0): The cross entropy loss.
        """
        return self.loss_fn(output, target)


class DiceLoss(nn.Module):
    """The Dice loss.
    Refs:
        https://arxiv.org/pdf/1606.04797
        https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895

    Args:
        binary (bool, optional): Whether to use the binary mode, which expects the task is
            binary classification, the activate function is sigmoid and output channel is 1
            (similar to torch.nn.BCEWithLogitsLoss and torch.nn.BCELoss) (default: False).
        with_logits (bool, optional): Specify the output is logits or probability (default: True).
        smooth (float, optional): The smooth term (default: 1.0).
        square (bool, optional): Whether to use the square of cardinality (default: True).
    """

    def __init__(self, binary=False, with_logits=True, smooth=1.0, square=True):
        super().__init__()
        self.binary = binary
        self.with_logits = with_logits
        self.smooth = smooth
        self.square = square

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *) or (N, 1, *): The output logits or probability.
            target (torch.LongTensor) (N, 1, *): The target where each value is between 0 and C-1.

        Returns:
            loss (torch.Tensor) (0): The dice loss.
        """
        if self.with_logits:
            output = F.sigmoid(output) if self.binary else F.softmax(output, dim=1)

        # Get the one-hot encoding of the ground truth label.
        if not self.binary:
            target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the dice loss.
        reduced_dims = tuple(range(2, output.dim()))  # (N, C, *) --> (N, C)
        intersection = (output * target).sum(reduced_dims)
        if self.square:
            cardinality = (output ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
        else:
            cardinality = output.sum(reduced_dims) + target.sum(reduced_dims)
        loss = (1 - (2 * intersection + self.smooth) / (cardinality + self.smooth)).mean()
        return loss


class TverskyLoss(nn.Module):
    """The Tversky loss.
    Refs:
        https://arxiv.org/abs/1706.05721

    Args:
        binary (bool, optional): Whether to use the binary mode, which expects the task is
            binary classification, the activate function is sigmoid and output channel is 1
            (similar to torch.nn.BCEWithLogitsLoss and torch.nn.BCELoss) (default: False).
        with_logits (bool, optional): Specify the output is logits or probability (default: True).
        alpha (float, optional): The magnitude of penalties for the False Positives (FPs) (default: 0.5).
        beta (float, optional): The magnitude of penalties for the False Negatives (FNs) (default: 0.5).
        smooth (float, optional): The smooth term (default: 1.0).
    """

    def __init__(self, binary=False, with_logits=True, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.binary = binary
        self.with_logits = with_logits
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *) or (N, 1, *): The output logits or probability.
            target (torch.LongTensor) (N, 1, *): The target where each value is between 0 and C-1.

        Returns:
            loss (torch.Tensor) (0): The tversky loss.
        """
        if self.with_logits:
            output = F.sigmoid(output) if self.binary else F.softmax(output, dim=1)

        # Get the one-hot encoding of the ground truth label.
        if not self.binary:
            target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the tversky loss.
        reduced_dims = tuple(range(2, output.dim()))  # (N, C, *) --> (N, C)
        intersection = (output * target).sum(reduced_dims)
        fps = (output * (1 - target)).sum(reduced_dims)
        fns = ((1 - output) * target).sum(reduced_dims)
        loss = (
            1 - (intersection + self.smooth) / (intersection + self.alpha * fps + self.beta * fns + self.smooth)
        ).mean()
        return loss
