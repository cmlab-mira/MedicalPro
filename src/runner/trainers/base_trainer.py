import importlib
import logging
import random
import torch
from torch.optim.lr_scheduler import (
    CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
)
from tqdm import tqdm


class BaseTrainer:
    """The base class for all trainers.
    Args:
        device (torch.device): The device.
        train_dataloader (Dataloader): The training dataloader.
        valid_dataloader (Dataloader): The validation dataloader.
        net (BaseNet): The network architecture.
        loss_fns (LossFns): The loss functions.
        loss_weights (torch.Tensor): The corresponding weights of loss functions.
        metric_fns (MetricFns): The metric functions.
        optimizer (torch.optim.Optimizer): The algorithm to train the network.
        lr_scheduler (torch.optim.lr_scheduler): The scheduler to adjust the learning rate.
        logger (BaseLogger): The object for recording the log information.
        monitor (Monitor): The object to determine whether to save the checkpoint.
        num_epochs (int): The total number of training epochs.
        valid_freq (int): The validation frequency (default: 1).
        use_amp (bool): Whether to use the Automatic Mixed Precision training (default: False).
        opt_level (str): The optimization level of apex.amp (default: 'O1').
    """

    def __init__(self, device, train_dataloader, valid_dataloader,
                 net, loss_fns, loss_weights, metric_fns, optimizer,
                 lr_scheduler, logger, monitor, num_epochs,
                 valid_freq=1, use_amp=False, opt_level='O1'):
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.net = net
        self.loss_fns = loss_fns
        self.loss_weights = loss_weights
        self.metric_fns = metric_fns
        self.optimizer = optimizer
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            raise ValueError(f'Do not support {ReduceLROnPlateau} scheduler yet.')
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.monitor = monitor
        self.num_epochs = num_epochs
        self.valid_freq = valid_freq
        self.use_amp = use_amp
        self.epoch = 1

        if use_amp:
            global amp
            amp = importlib.import_module('apex.amp')
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level=opt_level)

    def train(self):
        """The training process.
        """
        while self.epoch <= self.num_epochs:
            # Do training and validation.
            print()
            logging.info(f'Epoch {self.epoch}.')
            train_log, train_batch, train_outputs = self._run_epoch('train')
            logging.info(f'Train log: {train_log}.')
            if self.epoch % self.valid_freq == 0:
                valid_log, valid_batch, valid_outputs = self._run_epoch('valid')
                logging.info(f'Valid log: {valid_log}.')
            else:
                valid_log, valid_batch, valid_outputs = None, None, None

            # Adjust the learning rate.
            if (self.lr_scheduler is not None
                    and not isinstance(self.lr_scheduler,
                                       (CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts))):
                self.lr_scheduler.step()

            # Record the log information.
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
                    logging.info(f'Save the best checkpoint to {saved_path} '
                                 f'({self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')
                    self.save(saved_path)
                else:
                    epoch = self.epoch - self.monitor.not_improved_count * self.valid_freq
                    logging.info(f'The best checkpoint is remained at epoch {epoch} '
                                 f'({self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')

                # Early stop.
                if self.monitor.is_early_stopped():
                    logging.info('Early stopped.')
                    break

            self.epoch += 1

        self.logger.close()

    def _run_epoch(self, mode):
        """Run an epoch for training.
        Args:
            mode (str): The mode of running an epoch ('train' or 'valid').

        Returns:
            log (dict): The log information.
            batch (dict or sequence): The last batch of the data.
            outputs (torch.Tensor or sequence of torch.Tensor): The corresponding model outputs.
        """
        if mode == 'train':
            self.net.train()
            dataloader = self.train_dataloader
        else:
            self.net.eval()
            dataloader = self.valid_dataloader
        trange = tqdm(dataloader, total=len(dataloader), desc=mode)

        epoch_log = EpochLog()
        for i, batch in enumerate(trange):
            if mode == 'train':
                train_dict = self._train_step(batch)
                losses = train_dict['losses']
                _, _losses = zip(*sorted(losses.items()))
                loss = (torch.stack(_losses) * self.loss_weights).sum()
                metrics = train_dict.get('metrics')
                outputs = train_dict.get('outputs')
                if self.use_amp:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss /= dataloader.grad_accumulation_steps(i)
                        scaled_loss.backward()
                else:
                    (loss / dataloader.grad_accumulation_steps(i)).backward()
                if (i + 1) % dataloader.grad_accumulation_steps() == 0 or (i + 1) == len(dataloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if isinstance(self.lr_scheduler, (CyclicLR, OneCycleLR)):
                        self.lr_scheduler.step()
                    elif isinstance(self.lr_scheduler, CosineAnnealingWarmRestarts):
                        self.lr_scheduler.step((self.epoch - 1) + i / len(dataloader))
            else:
                with torch.no_grad():
                    valid_dict = self._valid_step(batch)
                    losses = valid_dict['losses']
                    _, _losses = zip(*sorted(losses.items()))
                    loss = (torch.stack(_losses) * self.loss_weights).sum()
                    metrics = valid_dict.get('metrics')
                    outputs = valid_dict.get('outputs')
            if (i + 1) == len(dataloader) and not dataloader.drop_last:
                batch_size = len(dataloader.dataset) % dataloader.batch_size
            else:
                batch_size = dataloader.batch_size
            epoch_log.update(batch_size, loss, losses, metrics)
            trange.set_postfix(**epoch_log.on_step_end_log)
        return epoch_log.on_epoch_end_log, batch, outputs

    def _train_step(self, batch):
        """The user-defined training logic.
        Args:
            batch (dict or sequence): A batch of the data.

        Returns:
            train_dict (dict): The computed results.
                train_dict['losses'] (dict)
                train_dict['metrics'] (dict, optional)
                train_dict['outputs'] (dict, optional)
        """
        raise NotImplementedError

    def _valid_step(self, batch):
        """The user-defined validation logic.
        Args:
            batch (dict or sequence): A batch of the data.

        Returns:
            valid_dict (dict): The computed results.
                valid_dict['losses'] (dict)
                valid_dict['metrics'] (dict, optional)
                valid_dict['outputs'] (dict, optional)
        """
        raise NotImplementedError

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
            'amp': amp.state_dict() if self.use_amp else None
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


class EpochLog:
    """The log to record the information of an epoch.
    """

    def __init__(self):
        self.count = 0
        self.log = None

    def _init_log(self, losses, metrics=None):
        """Initilize the log.
        Args:
            losses (dict): The computed losses.
            metrics (dict): The computed metrics.
        """
        self.log = {}
        self.log['loss'] = 0
        for loss_name in losses.keys():
            self.log[loss_name] = 0
        if metrics is not None:
            for metric_name in metrics.keys():
                self.log[metric_name] = 0

    def update(self, batch_size, loss, losses, metrics=None):
        """Accumulate the computed losses and metrics.
        Args:
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (dict): The computed losses.
            metrics (dict): The computed metrics.
            batch_size (int): The batch size.
        """
        self.count += batch_size
        if self.log is None:
            self._init_log(losses, metrics)
        self.log['loss'] += loss.item() * batch_size
        for loss_name, loss in losses.items():
            self.log[loss_name] += loss.item() * batch_size
        if metrics is not None:
            for metric_name, metric in metrics.items():
                self.log[metric_name] += metric.item() * batch_size

    @property
    def on_step_end_log(self):
        return dict((key, f'{value / self.count: .3f}') for key, value in self.log.items())

    @property
    def on_epoch_end_log(self):
        return dict((key, value / self.count) for key, value in self.log.items())
