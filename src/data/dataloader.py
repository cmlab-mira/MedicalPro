import torch
import numpy as np
from torch.utils.data import DataLoader


class Dataloader(DataLoader):
    """The modified class of ``torch.utils.data.DataLoader`` with default ``collate_fn`` and ``worker_init_fn``.
    Args:
        dataset (Dataset): Dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load (default: ``1``).
        shuffle (bool, optional): Set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
        sampler (Sampler, optional): Defines the strategy to draw samples from the dataset. If specified, ``shuffle`` must be False (default: ``None``).
        batch_sampler (Sampler, optional): Like ``sampler``, but returns a batch of indices at a time. Mutually exclusive with ``batch_size``, ``shuffle``, ``sampler``, and ``drop_last`` (default: ``None``).
        num_workers (int, optional): How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: ``0``)
        collate_fn (callable, optional): Merges a list of samples to form a mini-batch (default: ``default_collate``).
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors into CUDA pinned memory before returning them. If your data elements are a custom type, or your ``collate_fn`` returns a batch that is a custom type see the example below (default: ``False``).
        drop_last (bool, optional): Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then the last batch will be smaller (default: ``False``).
        timeout (numeric, optional): If positive, the timeout value for collecting a batch from workers. Should always be non-negative (default: ``0``).
        worker_init_fn (callable, optional): If not ``None``, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: ``_default_worker_init_fn``)
        grad_accumulation_steps (int): The number of gradient accumulation steps (default: 1).
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False,
                 drop_last=True, timeout=0, worker_init_fn=None, grad_accumulation_steps=1):
        if worker_init_fn is None:
            worker_init_fn = self._default_worker_init_fn

        if collate_fn is None:
            super().__init__(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             sampler=sampler,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             drop_last=drop_last,
                             timeout=timeout,
                             worker_init_fn=worker_init_fn)
        else:
            super().__init__(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             sampler=sampler,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             collate_fn=collate_fn,
                             pin_memory=pin_memory,
                             drop_last=drop_last,
                             timeout=timeout,
                             worker_init_fn=worker_init_fn)
        
        if dataset.type == 'train':
            self.grad_accumulation_steps = GradAccumulationSteps(grad_accumulation_steps=grad_accumulation_steps,
                                                                 drop_last=drop_last,
                                                                 total_steps=len(self._index_sampler),
                                                                 batch_size=batch_size,
                                                                 last_batch_size=len(dataset) % batch_size)
        else:
            self.grad_accumulation_steps = None
    
    def __len__(self):
        if self.drop_last and self.grad_accumulation_steps is not None and self.grad_accumulation_steps() != 1:
            return len(self._index_sampler) // self.grad_accumulation_steps() * self.grad_accumulation_steps()
        else:
            return len(self._index_sampler)
    
    @staticmethod
    def _default_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


class GradAccumulationSteps:
    """The gradient accumulation which considers the last training steps in an epoch.
    Args:
        grad_accumulation_steps (int): The number of gradient accumulation steps.
        drop_last (bool): Whether to drop the last incomplete batch.
        total_steps (int): The total number of training steps in an epoch.
        batch_size (int): How many samples per batch to load.
        last_batch_size (int): How many samples of the last incomplete batch.
    """
    def __init__(self, grad_accumulation_steps, drop_last, total_steps, batch_size, last_batch_size):
        self.grad_accumulation_steps = grad_accumulation_steps
        if drop_last or grad_accumulation_steps == 1:
            self.last_steps = {}
        else:
            if total_steps % grad_accumulation_steps == 0:
                indices = list(range(total_steps - grad_accumulation_steps, total_steps))
            else:
                indices = list(range(total_steps - total_steps % grad_accumulation_steps, total_steps))
            steps = np.array([batch_size] * (len(indices) - 1) + [last_batch_size])
            steps = steps.sum() / steps
            self.last_steps = {k: v for k, v in zip(indices, steps)}

    def __call__(self, i=None):
        if i in self.last_steps:
            return self.last_steps[i]
        else:
            return self.grad_accumulation_steps
