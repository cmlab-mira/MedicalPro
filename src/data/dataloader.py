import numpy as np
from torch.utils.data import DataLoader


class Dataloader(DataLoader):
    """The modified class of torch.utils.data.DataLoader.
    Args:
        dataset (BaseDataset): Dataset from which to load the data.
        grad_accumulation_steps (int): The number of gradient accumulation steps (default: 1).
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None, grad_accumulation_steps=1):
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
            if self.drop_last and grad_accumulation_steps > len(self._index_sampler):
                raise ValueError(f'The grad_accumulation_steps {grad_accumulation_steps} is greater than '
                                 f'the total_steps {len(self._index_sampler)} and the drop_last is true.')

            if len(dataset) % batch_size == 0:
                last_batch_size = batch_size
            else:
                last_batch_size = len(dataset) % batch_size
            self.grad_accumulation_steps = GradAccumulationSteps(grad_accumulation_steps=grad_accumulation_steps,
                                                                 drop_last=drop_last,
                                                                 total_steps=len(self._index_sampler),
                                                                 batch_size=batch_size,
                                                                 last_batch_size=last_batch_size)

    def __len__(self):
        if (getattr(self, 'grad_accumulation_steps', None) is not None
                and self.drop_last and self.grad_accumulation_steps() != 1):
            return len(self._index_sampler) // self.grad_accumulation_steps() * self.grad_accumulation_steps()
        else:
            return len(self._index_sampler)


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
            self.last_steps = {indice: step for indice, step in zip(indices, steps)}

    def __call__(self, i=None):
        if i in self.last_steps:
            return self.last_steps[i]
        else:
            return self.grad_accumulation_steps
