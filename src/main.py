import argparse
import logging
import random
import re
import torch
import yaml
from box import Box
from pathlib import Path

import src


def main(args):
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)
    saved_dir = Path(config.main.saved_dir)
    if not saved_dir.is_dir():
        saved_dir.mkdir(parents=True)

    logging.info(f'Save the config to "{config.main.saved_dir}".')
    with open(saved_dir / 'config.yaml', 'w+') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    if not args.test:
        random_seed = config.main.get('random_seed')
        if random_seed is None:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            # Make the experiment results deterministic.
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logging.info('Create the device.')
        if 'cuda' in config.trainer.kwargs.device and not torch.cuda.is_available():
            raise ValueError("The cuda is not available. Please set the device to 'cpu'.")
        device = torch.device(config.trainer.kwargs.device)

        logging.info('Create the training and validation datasets.')
        config.dataset.kwargs.update(type_='train')
        train_dataset = _get_instance(src.data.datasets, config.dataset)
        config.dataset.kwargs.update(type_='valid')
        valid_dataset = _get_instance(src.data.datasets, config.dataset)

        logging.info('Create the training and validation dataloaders.')
        cls = getattr(src.data.datasets, config.dataset.name)
        collate_fn = getattr(cls, 'collate_fn', None)
        train_batch_size = config.dataloader.kwargs.pop('train_batch_size')
        valid_batch_size = config.dataloader.kwargs.pop('valid_batch_size')
        config.dataloader.kwargs.update(collate_fn=collate_fn, batch_size=train_batch_size)
        train_dataloader = _get_instance(src.data.dataloader, config.dataloader, train_dataset)
        config.dataloader.kwargs.update(batch_size=valid_batch_size)
        valid_dataloader = _get_instance(src.data.dataloader, config.dataloader, valid_dataset)

        logging.info('Create the network architecture.')
        net = _get_instance(src.model.nets, config.net).to(device)

        logging.info('Create the loss functions and corresponding weights.')
        loss_fns, loss_weights = LossFns(), []
        defaulted_loss_fns = [loss_fn for loss_fn in dir(torch.nn) if 'Loss' in loss_fn]
        for config_loss in sorted(config.losses,
                                  key=lambda config_loss: _snake_case(config_loss.get('alias', config_loss.name))):
            if config_loss.name in defaulted_loss_fns:
                loss_fn = _get_instance(torch.nn, config_loss).to(device)
            else:
                loss_fn = _get_instance(src.model.losses, config_loss).to(device)
            loss_weight = config_loss.get('weight', 1 / len(config.losses))
            name = _snake_case(config_loss.get('alias', config_loss.name))
            setattr(loss_fns, name, loss_fn)
            loss_weights.append(loss_weight)
        loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)

        logging.info('Create the metric functions.')
        metric_fns = MetricFns()
        for config_metric in config.metrics:
            metric_fn = _get_instance(src.model.metrics, config_metric).to(device)
            name = _snake_case(config_metric.get('alias', config_metric.name))
            setattr(metric_fns, name, metric_fn)

        logging.info('Create the optimizer.')
        optimizer = _get_instance(torch.optim, config.optimizer, net.parameters())

        logging.info('Create the learning rate scheduler.')
        lr_scheduler = config.get('lr_scheduler')
        if lr_scheduler is not None:
            lr_scheduler = _get_instance(torch.optim.lr_scheduler, config.lr_scheduler, optimizer)

        logging.info('Create the logger.')
        config.logger.kwargs.update(log_dir=saved_dir / 'log',
                                    net=net)
        logger = _get_instance(src.callbacks.loggers, config.logger)

        logging.info('Create the monitor.')
        config.monitor.kwargs.update(checkpoints_dir=saved_dir / 'checkpoints')
        monitor = _get_instance(src.callbacks.monitor, config.monitor)

        logging.info('Create the trainer.')
        kwargs = {
            'device': device,
            'train_dataloader': train_dataloader,
            'valid_dataloader': valid_dataloader,
            'net': net,
            'loss_fns': loss_fns,
            'loss_weights': loss_weights,
            'metric_fns': metric_fns,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'logger': logger,
            'monitor': monitor
        }
        config.trainer.kwargs.update(kwargs)
        trainer = _get_instance(src.runner.trainers, config.trainer)

        loaded_path = config.main.get('loaded_path')
        if loaded_path is None:
            logging.info('Start training.')
        else:
            logging.info(f'Load the previous checkpoint from "{loaded_path}".')
            trainer.load(Path(loaded_path))
            logging.info('Resume training.')
        trainer.train()
        logging.info('End training.')
    else:
        logging.info('Create the device.')
        if 'cuda' in config.predictor.kwargs.device and not torch.cuda.is_available():
            raise ValueError("The cuda is not available. Please set the device to 'cpu'.")
        device = torch.device(config.predictor.kwargs.device)

        logging.info('Create the testing dataset.')
        config.dataset.kwargs.update(type_='test')
        test_dataset = _get_instance(src.data.datasets, config.dataset)

        logging.info('Create the testing dataloader.')
        test_dataloader = _get_instance(src.data.dataloader, config.dataloader, test_dataset)

        logging.info('Create the network architecture.')
        net = _get_instance(src.model.nets, config.net).to(device)

        logging.info('Create the loss functions and corresponding weights.')
        loss_fns, loss_weights = LossFns(), []
        defaulted_loss_fns = [loss_fn for loss_fn in dir(torch.nn) if 'Loss' in loss_fn]
        for config_loss in sorted(config.losses,
                                  key=lambda config_loss: _snake_case(config_loss.get('alias', config_loss.name))):
            if config_loss.name in defaulted_loss_fns:
                loss_fn = _get_instance(torch.nn, config_loss).to(device)
            else:
                loss_fn = _get_instance(src.model.losses, config_loss).to(device)
            loss_weight = config_loss.get('weight', 1 / len(config.losses))
            loss_name = _snake_case(config_loss.get('alias', config_loss.name))
            setattr(loss_fns, loss_name, loss_fn)
            loss_weights.append(loss_weight)
        loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)

        logging.info('Create the metric functions.')
        metric_fns = MetricFns()
        for config_metric in config.metrics:
            metric_fn = _get_instance(src.model.metrics, config_metric).to(device)
            metric_name = _snake_case(config_metric.get('alias', config_metric.name))
            setattr(metric_fns, metric_name, metric_fn)

        logging.info('Create the predictor.')
        kwargs = {
            'device': device,
            'test_dataloader': test_dataloader,
            'net': net,
            'loss_fns': loss_fns,
            'loss_weights': loss_weights,
            'metric_fns': metric_fns
        }
        config.predictor.kwargs.update(kwargs)
        predictor = _get_instance(src.runner.predictors, config.predictor)

        loaded_path = config.main.loaded_path
        logging.info(f'Load the previous checkpoint from "{loaded_path}".')
        predictor.load(Path(loaded_path))
        logging.info('Start testing.')
        predictor.predict()
        logging.info('End testing.')


class BaseFns:
    def __init__(self):
        pass

    def __getattr__(self, name):
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                             f"Its attributes: {list(self.__dict__.keys())}")


class LossFns(BaseFns):
    def __init__(self):
        super().__init__()


class MetricFns(BaseFns):
    def __init__(self):
        super().__init__()


def _parse_args():
    parser = argparse.ArgumentParser(description="The main pipeline script.")
    parser.add_argument('config_path', type=Path, help='The path of the config file.')
    parser.add_argument('--test', action='store_true',
                        help='Perform testing if specified; otherwise perform training.')
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
    return cls(*args) if kwargs is None else cls(*args, **config.kwargs)


def _snake_case(string):
    """Convert a string into snake case form.
    Ref: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    return re.sub('((?<=[a-z0-9])[A-Z]|(?!^)(?<!_)[A-Z](?=[a-z]))', r'_\1', string).lower()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
