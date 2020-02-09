import torch
import torch.nn as nn

__all__ = [
    'Dice',
    'Accuracy',
]


class Dice(nn.Module):
    """The Dice score.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.

        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.
        pred = output.argmax(dim=1, keepdim=True)
        pred = torch.zeros_like(output).scatter_(1, pred, 1)
        target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the dice score.
        reduced_dims = list(range(2, output.dim()))  # (N, C, *) --> (N, C)
        intersection = (pred * target).sum(reduced_dims)
        cardinality = pred.sum(reduced_dims) + target.sum(reduced_dims)
        score = (2 * intersection / cardinality.clamp(min=1)).mean(dim=0)
        return score

    
class Accuracy(nn.Module):
    """The accuracy of the classification task.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        """
        Args:
            logits (torch.tensor) (N, C): The model output logits.
            labels (torch.tensor) (N, C): The data labels.

        Returns:
            score (torch.tensor) (0): The accuracy score.
        """
        preds = logits.argmax(dim=1)
        labels = labels.squeeze(dim=1)
        correct = (preds == labels).sum()
        return correct / logits.size(0)