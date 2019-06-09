import argparse
import logging
import ipdb
import sys
import torch
import random
import importlib
import yaml
from box import Box
from pathlib import Path

import src


def main(args):
    logging.info(f'Load the config from {args.config_path} and save it to {config.main.saved_dir}.')
    config = Box.from_yaml(filename=args.config_path)
    with open(Path(config.main.saved_dir) / 'config.yaml') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Make the experiment results deterministic.
    random.seed(config.main.random_seed)
    torch.manual_seed(random.getstate()[1][1])
    torch.cuda.manual_seed_all(random.getstate()[1][1])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info('Create the device.')
    if 'cuda' in config.trainer.device and not torch.cuda.is_available():
        raise ValueError("The cuda is not available. Please set the device in the trainer section to 'cpu'.")
    device = torch.device(config.trainer.device)

    logging.info('Create the training and validation datasets.')
    train_dataset = _get_instance(src.data.datasets, config.dataset, 'train')
    valid_dataset = _get_instance(src.data.datasets, config.dataset, 'valid')

    logging.info('Create the training and validation dataloaders.')
    cls = getattr(src.data.datasets, config.dataset.name)
    config.dataloader.kwargs.update(collate_fn=getattr(cls, 'collate_fn', None))
    train_dataloader = _get_instance(src.data.dataloader, config.dataloader, train_dataset)
    valid_dataloader = _get_instance(src.data.dataloader, config.dataloader, valid_dataset)

    logging.info('Create the network architecture.')
    net = _get_instance(src.model.nets, config.net)

    logging.info('Create the loss functions and the corresponding weights.')
    losses, loss_weights = [], []
    defaulted_losses = [loss for loss in dir(torch.nn) if 'Loss' in loss]
    for config_loss in config.losses:
        if config_loss.name in defaulted_losses:
            loss = _get_instance(torch.nn, config_loss)
        else:
            loss = _get_instance(src.model.losses, config_loss)
        losses.append(loss)
        loss_weights.append(config_loss.weight)

    logging.info('Create the metric function.')
    metrics = [_get_instance(src.model.metrics, config_metric) for config_metric in config.metrics]

    logging.info('Create the optimizer.')
    optimizer = _get_instance(torch.optim, config.optimizer, net.parameters())

    logging.info('Create the learning rate scheduler.')
    lr_scheduler = _get_instance(torch.optim.lr_scheduler, config.lr_scheduler, optimizer) if config.get('lr_scheduler') else None

    logging.info('Create the logger.')
    config.logger.kwargs.update(log_dir=config.main.saved_dir + '/log', net=net)
    logger = _get_instance(src.callbacks.loggers, config.logger)

    logging.info('Create the monitor.')
    config.monitor.kwargs.update(root=config.main.saved_dir + '/checkpoints')
    monitor = _get_instance(src.callbacks.monitor, config.monitor)

    logging.info('Create the trainer.')
    kwargs = {'device': device,
              'train_dataloader': train_dataloader,
              'valid_dataloader': valid_dataloader,
              'net': net,
              'losses': losses,
              'loss_weights': loss_weights,
              'metrics': metrics,
              'optimizer': optimizer,
              'lr_scheduler': lr_scheduler,
              'logger': logger,
              'monitor': monitor}
    config.trainer.kwargs.update(kwargs)
    trainer = _get_instance(src.runner.trainers, config.trainer)

    # Start training.
    logging.info('Start training.')
    trainer.train()
    logging.info('End training.')


def _parse_args():
    parser = argparse.ArgumentParser(description="The script to train.")
    parser.add_argument('config_file', type=Path, help='The path of the config file.')
    args = parser.parse_args()
    return args


def _get_instance(module, config, *args):
    """
    Args:
        module (module): The python module.
        config (Box): The config to create the class object.

    Returns:
        instance (object): The class object defined in the module.
    """
    cls = getattr(module, config.name)
    kwargs = config.get('kwargs')
    return cls(*args, **config.kwargs) if kwargs else cls(*args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        args = _parse_args()
        main(args)
