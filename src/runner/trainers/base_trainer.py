import logging
import random
import torch
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    logging.warning(f'The apex.amp is not available!')


class BaseTrainer:
    """The base class for all trainers.
    Args:
        device (torch.device): The device.
        train_dataloader (Dataloader): The training dataloader.
        valid_dataloader (Dataloader): The validation dataloader.
        net (BaseNet): The network architecture.
        loss_fns (list of torch.nn.Module): The loss functions.
        loss_weights (list of float): The corresponding weights of loss functions.
        metric_fns (list of torch.nn.Module): The metric functions.
        optimizer (torch.optim.Optimizer): The algorithm to train the network.
        lr_scheduler (torch.optim._LRScheduler): The scheduler to adjust the learning rate.
        logger (BaseLogger): The object for recording the log information and visualization.
        monitor (Monitor): The object to determine whether to save the checkpoint.
        num_epochs (int): The total number of training epochs.
        valid_freq (int): The validation frequency (default: 1).
        opt_level (str): The optimization level of apex.amp (default: 'O0').
    """
    def __init__(self, device, train_dataloader, valid_dataloader,
                 net, loss_fns, loss_weights, metric_fns, optimizer,
                 lr_scheduler, logger, monitor, num_epochs,
                 valid_freq=1, opt_level='O0'):
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.net = net.to(device)
        self.loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)
        self.metric_fns = [metric_fn.to(device) for metric_fn in metric_fns]
        self.optimizer = optimizer
        if APEX_AVAILABLE:
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level=opt_level)
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise NotImplementedError('Do not support torch.optim.lr_scheduler.ReduceLROnPlateau scheduler yet.')
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.monitor = monitor
        self.num_epochs = num_epochs
        self.valid_freq = valid_freq
        self.epoch = 1

    def train(self):
        """The training process.
        """
        while self.epoch <= self.num_epochs:
            # Do training and validation.
            print()
            logging.info(f'Epoch {self.epoch}.')
            train_log, train_batch, train_outputs = self._run_epoch('training')
            logging.info(f'Train log: {train_log}.')
            if self.epoch % self.valid_freq == 0:
                valid_log, valid_batch, valid_outputs = self._run_epoch('validation')
                logging.info(f'Valid log: {valid_log}.')
            else:
                valid_log, valid_batch, valid_outputs = None, None, None

            # Adjust the learning rate.
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, (CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts)):
                self.lr_scheduler.step()

            # Record the log information and visualization.
            self.logger.write(self.epoch, train_log, train_batch, train_outputs,
                              valid_log, valid_batch, valid_outputs)

            # Save the regular checkpoint.
            saved_path = self.monitor.is_saved(self.epoch)
            if saved_path:
                logging.info(f'Save the checkpoint to {saved_path}.')
                self.save(saved_path)

            if self.epoch % self.valid_freq == 0:
                # Save the best checkpoint.
                saved_path = self.monitor.is_best(valid_log)
                if saved_path:
                    logging.info(f'Save the best checkpoint to {saved_path} ({self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')
                    self.save(saved_path)
                else:
                    logging.info(f'The best checkpoint is remained (at epoch {self.epoch - self.monitor.not_improved_count * self.valid_freq}, {self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')

                # Early stop.
                if self.monitor.is_early_stopped():
                    logging.info('Early stopped.')
                    break

            self.epoch +=1

        self.logger.close()

    def _run_epoch(self, mode):
        """Run an epoch for training.
        Args:
            mode (str): The mode of running an epoch ('training' or 'validation').

        Returns:
            log (dict): The log information.
            batch (dict or sequence): The last batch of the data.
            outputs (torch.Tensor or sequence of torch.Tensor): The corresponding model outputs.
        """
        if mode == 'training':
            self.net.train()
        else:
            self.net.eval()
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(enumerate(dataloader),
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()
        count = 0
        for i, batch in trange:
            batch = self._allocate_data(batch)
            inputs, targets = self._get_inputs_targets(batch)
            if mode == 'training':
                outputs = self.net(inputs)
                losses = self._compute_losses(outputs, targets)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                if APEX_AVAILABLE:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss /= dataloader.grad_accumulation_steps(i)
                        scaled_loss.backward()
                else:
                    loss /= dataloader.grad_accumulation_steps(i)
                    loss.backward()
                if (i + 1) % dataloader.grad_accumulation_steps() == 0 or (i + 1) == len(dataloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if isinstance(self.lr_scheduler, (CyclicLR, OneCycleLR)):
                        self.lr_scheduler.step()
                    elif isinstance(self.lr_scheduler, CosineAnnealingWarmRestarts):
                        self.lr_scheduler.step((self.epoch - 1) + i / len(dataloader))
                with torch.no_grad():
                    metrics = self._compute_metrics(outputs, targets)
            else:
                with torch.no_grad():
                    outputs = self.net(inputs)
                    losses = self._compute_losses(outputs, targets)
                    loss = (torch.stack(losses) * self.loss_weights).sum()
                    metrics =  self._compute_metrics(outputs, targets)

            batch_size = self.train_dataloader.batch_size if mode == 'training' else self.valid_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        for key in log:
            log[key] /= count
        return log, batch, outputs

    def _allocate_data(self, batch):
        """Allocate the data to the device.
        Args:
            batch (dict or sequence): A batch of the data.

        Returns:
            batch (dict or sequence): A batch of the allocated data.
        """
        if isinstance(batch, dict):
            return dict((key, self._allocate_data(data)) for key, data in batch.items())
        elif isinstance(batch, list):
            return list(self._allocate_data(data) for data in batch)
        elif isinstance(batch, tuple):
            return tuple(self._allocate_data(data) for data in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)

    def _get_inputs_targets(self, batch):
        """Specify the data inputs and targets.
        Args:
            batch (dict or sequence): A batch of data.

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
        for loss_fn in self.loss_fns:
            log[loss_fn.__class__.__name__] = 0
        for metric_fn in self.metric_fns:
            log[metric_fn.__class__.__name__] = 0
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
        for loss_fn, loss in zip(self.loss_fns, losses):
            log[loss_fn.__class__.__name__] += loss.item() * batch_size
        for metric_fn, metric in zip(self.metric_fns, metrics):
            log[metric_fn.__class__.__name__] += metric.item() * batch_size

    def save(self, path):
        """Save the model checkpoint.
        Args:
            path (Path): The path to save the model checkpoint.
        """
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'monitor': self.monitor.state_dict(),
            'epoch': self.epoch,
            'random_state': random.getstate(),
            'amp': amp.state_dict() if APEX_AVAILABLE else None
        }, path)

    def load(self, path):
        """Load the model checkpoint.
        Args:
            path (Path): The path to load the model checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['lr_scheduler'] is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.monitor.load_state_dict(checkpoint['monitor'])
        self.epoch = checkpoint['epoch'] + 1
        random.setstate(checkpoint['random_state'])
        if checkpoint['amp'] is not None:
            amp.load_state_dict(checkpoint['amp'])
