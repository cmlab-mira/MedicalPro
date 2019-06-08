import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from pathlib import Path

from src.model.nets.base_net import BaseNet
from src.runner.trainers.base_trainer import BaseTrainer
from src.callbacks.loggers.base_logger import BaseLogger
from src.callbacks.monitor import Monitor


class MyNet(BaseNet):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MyMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, batch):
        _, target = batch
        pred = output.argmax(dim=1, keepdim=True)
        return pred.eq(target.view_as(pred)).sum() / output.size(0)


class MyLogger(BaseLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output):
        data, target = valid_batch
        self.writer.add_images('data', data, epoch)


class MyTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run_iter(self, batch):
        data, target = batch
        output = self.net(data)
        losses = tuple(loss(output, target) for loss in self.losses)
        return output, losses


def test_trainer(tmpdir):
    nets = []
    root = tmpdir.mkdir('./data')
    for i in range(2):
        random.seed('MNIST')
        torch.manual_seed(random.getstate()[1][1])
        torch.cuda.manual_seed_all(random.getstate()[1][1])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        kwargs = {'batch_size': 64, 'shuffle': True, 'num_workers': 1}
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_dataloader = DataLoader(datasets.MNIST(root=root, train=True, download=True, transform=transform), **kwargs)
        valid_dataloader = DataLoader(datasets.MNIST(root=root, train=False, download=True, transform=transform), **kwargs)

        net = MyNet()
        losses = [nn.NLLLoss()]
        loss_weights = [1.0]
        metrics = [MyMetric()]
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
        lr_scheduler = None

        logger = MyLogger(log_dir='./models', net=net, dummy_input=torch.randn(1, 1, 28, 28))
        monitor = Monitor(root=Path('./models'), mode='min', target='Loss', saved_freq=5, early_stop=0)
        num_epochs = 10

        trainer = MyTrainer(device=device, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, net=net, losses=losses, loss_weights=loss_weights, metrics=metrics, optimizer=optimizer, lr_scheduler=lr_scheduler, logger=logger, monitor=monitor, num_epochs=num_epochs)

        trainer.train()

        nets.append(trainer.net)

    for params0, params1 in zip(nets[0].parameters(), nets[1].parameters()):
        assert torch.equal(params0, params1)
