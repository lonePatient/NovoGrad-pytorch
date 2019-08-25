import torch
import argparse
import torch.nn as nn
from nn.vgg import vgg16_bn
from nn.alexnet import alexnet
from nn.resnet import ResNet18
from tools import AverageMeter
from progressbar import ProgressBar
from tools import seed_everything
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from trainingmonitor import TrainingMonitor
from optimizer import NovoGrad,AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

epochs = 30
batch_size = 200
seed = 42

seed_everything(seed)
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cuda:0")

def train(train_loader,lr_scheduler = None):
    pbar = ProgressBar(n_batch=len(train_loader))
    train_loss = AverageMeter()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        pbar.batch_step(batch_idx = batch_idx,info = {'loss':loss.item()},bar_type='Training')
        train_loss.update(loss.item(),n =1)
    return {'loss':train_loss.avg}

def test(test_loader):
    pbar = ProgressBar(n_batch=len(test_loader))
    valid_loss = AverageMeter()
    valid_acc = AverageMeter()
    model.eval()
    count = 0
    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            valid_loss.update(loss,n = data.size(0))
            valid_acc.update(correct, n=1)
            count += data.size(0)
            pbar.batch_step(batch_idx=batch_idx, info={}, bar_type='Testing')
    return {'valid_loss':valid_loss.avg,
            'valid_acc':valid_acc.sum /count}

data = {
    'train': datasets.CIFAR10(
        root='./data', download=True,
        transform=transforms.Compose([
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))]
        )
    ),
    'valid': datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))]
        )
    )
}

loaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True,
                        num_workers=10, pin_memory=True,
                        drop_last=True),
    'valid': DataLoader(data['valid'], batch_size = batch_size,
                        num_workers=10, pin_memory=True,
                        drop_last=False)
}
if __name__  == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10')
    parser.add_argument("--model", type=str, default='alexnet',choices=['alexnet','vgg','resnet'])
    parser.add_argument("--task", type=str, default='Image')
    parser.add_argument("--optimizer", default='adamw', choices=['adamw', 'adam','novograd'])
    parser.add_argument('--do_scheduler',action='store_true')
    args = parser.parse_args()

    if args.model == 'alexnet':
        model = alexnet(num_classes=10)
    if args.model == 'vgg':
        model = vgg16_bn(num_classes=10)
    if args.model == 'resnet':
        model = ResNet18(num_classes=10)

    args.model += f"_{args.optimizer}"
    if args.do_scheduler:
        args.model += "_cosine"

    model.to(device)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    if args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=0.001)

    if args.optimizer == 'novograd':
        optimizer = NovoGrad(model.parameters(), lr=0.01, betas=(0.95, 0.98), weight_decay=0.001)

    train_monitor = TrainingMonitor(file_dir='./png', arch=args.model)
    if args.do_scheduler:
        lr_scheduler = CosineAnnealingLR(optimizer,epochs * len(loaders['train']),1e-4)

    for epoch in range(1, epochs + 1):
        if args.do_scheduler:
            train_log = train(loaders['train'],lr_scheduler = lr_scheduler)
        else:
            train_log = train(loaders['train'], lr_scheduler=None)
        valid_log = test(loaders['valid'])
        logs = dict(train_log, **valid_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        print(show_info)
        train_monitor.epoch_step(logs)
