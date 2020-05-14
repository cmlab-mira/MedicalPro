import torch
import torch.nn as nn

__all__ = [
    'Dice',
]


class Dice(nn.Module):
    """The Dice score.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *) or (N, 1, *): The output probability.
            target (torch.LongTensor) (N, 1, *): The target where each value is between 0 and C-1.

        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.
        if output.size(1) == 1:  # (N, 1, *) --> (N, 2, *)
            pred = torch.cat((1 - output, output), dim=1)
        _, pred = output.max(dim=1, keepdim=True)
        pred = torch.zeros_like(output).scatter_(1, pred, 1)
        target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the dice scores for each class.
        reduced_dims = tuple(range(2, output.dim()))  # (N, C, *) --> (N, C)
        intersection = (pred * target).sum(reduced_dims)
        cardinality = pred.sum(reduced_dims) + target.sum(reduced_dims)
        score = (2 * intersection / cardinality.clamp(min=1)).mean(dim=0)
        return score
