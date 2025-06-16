#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# from bb import get_arcode
import simsiam.loader
import simsiam.builder
from simsiam.loader import ROIDataset, get_label
from simsiam.models import Classifier
import json

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')
def gettrain_val(path):
    data = dict()
    classes = (0, 1, 2, 3, 4, 5, 6)
    for label in classes:
        data[label] = []
    for i, file in enumerate(os.listdir(path)):
        with open(os.path.join(path, file)) as json_file:
            roi = json.load(json_file)
            roi['name'] = json_file.name
            if roi['label'] in classes:
                data[roi['label']].append(roi)
    random.shuffle(data[0])
    random.shuffle(data[1])
    random.shuffle(data[2])
    random.shuffle(data[3])
    random.shuffle(data[4])
    random.shuffle(data[5])
    random.shuffle(data[6])
    train_name=[]
    val_name=[]
    for j in range(len(data[0])):
        if j<int(len(data[0])*95/100):
            train_name.append(data[0][j]['name'])
        else:
            val_name.append(data[0][j]['name'])
    for j in range(len(data[1])):
        if j<int(len(data[1])*95/100):
            train_name.append(data[1][j]['name'])
        else:
            val_name.append(data[1][j]['name'])
    for j in range(len(data[2])):
        if j<int(len(data[2])*95/100):
            train_name.append(data[2][j]['name'])
        else:
            val_name.append(data[2][j]['name'])
    for j in range(len(data[3])):
        if j<int(len(data[3])*95/100):
            train_name.append(data[3][j]['name'])
        else:
            val_name.append(data[3][j]['name'])
    for j in range(len(data[4])):
        if j<int(len(data[4])*95/100):
            train_name.append(data[4][j]['name'])
        else:
            val_name.append(data[4][j]['name'])
    for j in range(len(data[5])):
        if j<int(len(data[5])*95/100):
            train_name.append(data[5][j]['name'])
        else:
            val_name.append(data[5][j]['name'])
    for j in range(len(data[6])):
        if j<int(len(data[6])*95/100):
            train_name.append(data[6][j]['name'])
        else:
            val_name.append(data[6][j]['name'])
    random.shuffle(train_name)
    random.shuffle(val_name)
    return train_name,val_name
def main():
    # args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Current device is {}'.format(device))
    # create model
    print("=> creating model ...")
    model = Classifier().to(device)

    model.cuda()
    # infer learning rate before changing batch size
    batch_size = 256
    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1)
    # optimizer = torch.optim.SGD(optim_params, init_lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    # augmentation = [
    #     transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    #     transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize
    # ]
    train_name = []
    val_name = []
    train_name, val_name = gettrain_val(r'data/train_set_CNN')
    train_dataset = ROIDataset(path=r'data/train_set_CNN', load=train_name, key=get_label,
                               mode='classification', gen_p=0)
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # train_dataset, label1, X_dev, label2, X_test, label3 = get_arcode()
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size)

    for epoch in range(100):
        train(train_loader, model, criterion, optimizer, epoch,device)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))

def main_worker(gpu, ngpus_per_node, args):



    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         if args.gpu is None:
    #             checkpoint = torch.load(args.resume)
    #         else:
    #             # Map model  to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(args.gpu)
    #             checkpoint = torch.load(args.resume, map_location=loc)
    #         args.start_epoch = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
    #
    # cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    train_dataset, label1, X_dev, label2, X_test, label3 = get_arcode()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)

        if (epoch/100 == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch,device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i,(x, x1) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        x = x.to(device)
        x1= x1.to(device)

        # compute output and loss
        p1, p2, z1, z2 = model(x, x1)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
