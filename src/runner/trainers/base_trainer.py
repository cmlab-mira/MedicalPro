import torch
import logging
from tqdm import tqdm
import random
import numpy as np


class BaseTrainer:
    """The base class for all trainers.
    Args:
        device (torch.device): The device.
        train_dataloader (Dataloader): The training dataloader.
        valid_dataloader (Dataloader): The validation dataloader.
        net (BaseNet): The network architecture.
        losses (list of torch.nn.Module): The loss functions.
        loss_weights (list of float): The corresponding weights of loss functions.
        metrics (list of torch.nn.Module): The metric functions.
        optimizer (torch.optim.Optimizer): The algorithm to train the network.
        scheduler (torch.optim._LRScheduler): The scheduler to adjust the learning rate.
        logger (Logger): The object for recording the log information and visualization.
        monitor (Monitor): The object to determine whether to save the checkpoint.
        num_epochs (int): The total number of training epochs.
    """
    def __init__(self, device, train_dataloader, valid_dataloader,
                 net, losses, loss_weights, metrics, optimizer,
                 scheduler, logger, monitor, num_epochs):
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.net = net.to(device)
        self.losses = [loss.to(device) for loss in losses]
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)
        self.metrics = [metric.to(device) for metric in metrics]
        self.optimizer = optimizer

        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            raise NotImplementedError('Do not support torch.optim.lr_scheduler.CyclicLR scheduler yet.')
        self.scheduler = scheduler

        self.logger = logger
        self.monitor = monitor
        self.num_epochs = num_epochs
        self.epoch = 1

    def train(self):
        """The training process.
        """
        random_seeds = random.sample(range(10000000), k=self.num_epochs)
        while self.epoch <= self.num_epochs:
            # Reset the numpy random seed.
            np.random.seed(random_seeds[self.epoch])

            logging.info(f'\nEpoch {self.epoch}.')
            train_log, train_batch, train_output = self._run_epoch('training')
            logging.info(f'Train log: {train_log}.')
            valid_log, valid_batch, valid_output = self._run_epoch('validation')
            logging.info(f'Valid log: {valid_log}.')

            # Record the log information and visualization.
            self.logger.write(self.epoch, train_log, train_batch, train_output,
                              valid_log, valid_batch, valid_output)

            # Save the regular checkpoint.
            if self.monitor.is_saved(self.epoch):
                path = self.monitor.root / f'model_{self.epoch}'
                logging.info(f'Save the checkpoint to {path}.')
                self.save(path)

            # Save the best checkpoint.
            if self.monitor.is_best(valid_log):
                path = self.monitor.root / 'model_best'
                logging.info(f'Save the best checkpoint to {path} ({self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')
                self.save(path)
            else:
                logging.info(f'The best checkpoint is remained (at epoch {self.epoch - self.monitor.not_imporved_count}, {self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')

            # Early stop.
            if self.monitor.is_early_stopped():
                logging.info('Early stopped.')
                break

            self.epoch +=1

    def _run_epoch(self, mode):
        """Run an epoch for training.
        Args:
            mode (str): The mode of running an epoch ('training' or 'validation').

        Returns:
            log (dict): The log information.
            batch (dict or tuple): The last batch of the data.
            output (torch.Tensor): The corresponding output.
        """
        if mode == 'training':
            self.net.train()
        else:
            self.net.eval()
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()
        count = 0
        for batch in trange:
            if isinstance(batch, dict):
                batch = dict((key, data.to(self.device)) for key, data in batch.items())
            else:
                batch = tuple(data.to(self.device) for data in batch)

            if mode == 'training':
                output, *losses = self._run_iter(batch)
                loss = (torch.cat(losses) * self.loss_weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grads():
                    output, *losses = self._run_iter(batch)
                    loss = (torch.cat(losses) * self.loss_weights).sum()

            if self.scheduler is None:
                pass
            elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and mode == 'validation':
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

            batch_size = output.size(0)
            log['Loss'] += loss.item() * batch_size
            for loss, _loss in zip(self.losses, losses):
                log[loss.__class__.__name__] += _loss.item() * batch_size
            for metric in self.metrics:
                score = metric(output, batch)
                log[metric.__class__.__name__] += score.item() * batch_size
            count += batch_size
            trange.set_postfix(**log)

        for key in log:
            log[key] /= count
        return log, batch, output

    def _run_iter(self, batch):
        """Run an iteration to obtain the output and the losses.
        Args:
            batch (dict or tuple): A batch of data.

        Returns:
            output (torch.Tensor): The computed output.
            losses (sequence of torch.Tensor): The computed losses.
        """
        raise NotImplementedError

    def _init_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {}
        log['Loss'] = 0
        for loss in self.losses:
            log[loss.__class__.__name__] = 0
        for metric in self.metrics:
            log[metric.__class__.__name__] = 0
        return log

    def save(self, path):
        """Save the model checkpoint.
        Args:
            path (Path): The path to save the model checkpoint.
        """
        torch.save({
            'epoch': self.epoch,
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'monitor': self.monitor
        }, path)

    def load(self, path):
        """Load the model checkpoint.
        Args:
            path (Path): The path to load the model checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.epoch = checkpoint['epoch'] + 1
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.monitor = checkpoint['monitor']
