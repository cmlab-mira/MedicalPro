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
        lr_scheduler (torch.optim._LRScheduler): The scheduler to adjust the learning rate.
        logger (Logger): The object for recording the log information and visualization.
        monitor (Monitor): The object to determine whether to save the checkpoint.
        num_epochs (int): The total number of training epochs.
    """
    def __init__(self, device, train_dataloader, valid_dataloader,
                 net, losses, loss_weights, metrics, optimizer,
                 lr_scheduler, logger, monitor, num_epochs):
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.net = net.to(device)
        self.losses = [loss.to(device) for loss in losses]
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)
        self.metrics = [metric.to(device) for metric in metrics]
        self.optimizer = optimizer

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.CyclicLR):
            raise NotImplementedError('Do not support torch.optim.lr_scheduler.CyclicLR scheduler yet.')
        self.lr_scheduler = lr_scheduler

        self.logger = logger
        self.monitor = monitor
        self.num_epochs = num_epochs
        self.epoch = 1
        self.np_random_seeds = None

    def train(self):
        """The training process.
        """
        if self.np_random_seeds is None:
            self.np_random_seeds = random.sample(range(10000000), k=self.num_epochs)
        while self.epoch <= self.num_epochs:
            # Reset the numpy random seed.
            np.random.seed(self.np_random_seeds[self.epoch - 1])

            print()
            logging.info(f'Epoch {self.epoch}.')
            train_log, train_batch, train_outputs = self._run_epoch('training')
            logging.info(f'Train log: {train_log}.')
            valid_log, valid_batch, valid_outputs = self._run_epoch('validation')
            logging.info(f'Valid log: {valid_log}.')

            # Record the log information and visualization.
            self.logger.write(self.epoch, train_log, train_batch, train_outputs,
                              valid_log, valid_batch, valid_outputs)

            # Save the regular checkpoint.
            saved_path = self.monitor.is_saved(self.epoch)
            if saved_path:
                logging.info(f'Save the checkpoint to {saved_path}.')
                self.save(saved_path)

            # Save the best checkpoint.
            saved_path = self.monitor.is_best(valid_log)
            if saved_path:
                logging.info(f'Save the best checkpoint to {saved_path} ({self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')
                self.save(saved_path)
            else:
                logging.info(f'The best checkpoint is remained (at epoch {self.epoch - self.monitor.not_improved_count}, {self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')

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
            outputs (torch.Tensor or sequence of torch.Tensor): The corresponding model outputs.
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

            inputs, targets = self._set_inputs_targets(batch)
            if mode == 'training':
                outputs = self.net(inputs)
                losses = self._compute_losses(outputs, targets)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.net(inputs)
                    losses = self._compute_losses(outputs, targets)
                    loss = (torch.stack(losses) * self.loss_weights).sum()
            metrics =  self._compute_metrics(outputs, targets)

            if self.lr_scheduler is None:
                pass
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and mode == 'validation':
                self.lr_scheduler.step(loss)
            else:
                self.lr_scheduler.step()

            batch_size = self.train_dataloader.batch_size if mode == 'training' else self.valid_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        for key in log:
            log[key] /= count
        return log, batch, outputs

    def _set_inputs_targets(self, batch):
        """Specify the data inputs and targets.
        Args:
            batch (dict or tuple): A batch of data.

        Returns:
            inputs (torch.Tensor or sequence of torch.Tensor): The data inputs.
            targets (torch.Tensor or sequence of torch.Tensor): The data targets.
        """
        raise NotImplementedError

    def _compute_losses(self, outputs, targets):
        """Compute the losses.
        Args:
            outputs (torch.Tensor or sequence of torch.Tensor): The model outputs.
            targets (torch.Tensor or sequence of torch.Tensor): The data targets.

        Returns:
            losses (sequence of torch.Tensor): The computed losses.
        """
        raise NotImplementedError

    def _compute_metrics(self, outputs, targets):
        """Compute the metrics.
        Args:
            outputs (torch.Tensor or sequence of torch.Tensor): The model outputs.
            targets (torch.Tensor or sequence of torch.Tensor): The data targets.

        Returns:
            metrics (sequence of torch.Tensor): The computed metrics.
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

    def _update_log(self, log, batch_size, loss, losses, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            batch_size (int): The batch size.
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (sequence of torch.Tensor): The computed losses.
            metrics (sequence of torch.Tensor): The computed metrics.
        """
        log['Loss'] += loss.item() * batch_size
        for loss, _loss in zip(self.losses, losses):
            log[loss.__class__.__name__] += _loss.item() * batch_size
        for metric, _metric in zip(self.metrics, metrics):
            log[metric.__class__.__name__] += _metric.item() * batch_size

    def save(self, path):
        """Save the model checkpoint.
        Args:
            path (Path): The path to save the model checkpoint.
        """
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'monitor': self.monitor,
            'epoch': self.epoch,
            'random_state': random.getstate(),
            'np_random_seeds': self.np_random_seeds
        }, path)

    def load(self, path):
        """Load the model checkpoint.
        Args:
            path (Path): The path to load the model checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['lr_scheduler']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.monitor = checkpoint['monitor']
        self.epoch = checkpoint['epoch'] + 1
        random.setstate(checkpoint['random_state'])
        self.np_random_seeds = checkpoint['np_random_seeds']
