
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
from simsiam.loader import ROIDataset1, get_label
from simsiam.models import Classifier
import json
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch import Tensor
from torch.nn import functional as F
class FocalLoss(nn.Module):
    def __init__(self, class_num=7, alpha=None, gamma=1, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        '''
        inputs: shape [N,C]
        targets:shape [N]
        '''
        P = F.softmax(inputs)
        ids=targets.view(-1,1)
        class_mask=torch.zeros_like(inputs)
        class_mask.scatter_(dim=1, index=ids, value=1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        # batch_loss = -alpha * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
def gettest(path):
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
    test_name = []
    for j in range(len(data[0])):
        test_name.append(data[0][j]['name'])
    for j in range(len(data[1])):
        test_name.append(data[1][j]['name'])
    for j in range(len(data[2])):
        test_name.append(data[2][j]['name'])
    for j in range(len(data[3])):
        test_name.append(data[3][j]['name'])
    for j in range(len(data[4])):
        test_name.append(data[4][j]['name'])
    for j in range(len(data[5])):
        test_name.append(data[5][j]['name'])
    for j in range(len(data[6])):
        test_name.append(data[6][j]['name'])
    random.shuffle(test_name)
    return test_name
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
best_acc1 = 0


def main():
    # args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Current device is {}'.format(device))
    # create model
    print("=> creating model ...")
    model = Classifier().to(device)
    global best_acc1
    print("=> loading checkpoint ...")
    checkpoint = torch.load('checkpoint_0030.pth.tar', map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict, strict=False)


    print("=> loaded pre-trained model ...")
    model.cuda()
    # infer learning rate before changing batch size
    batch_size = 256
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss()
    # optimizer = torch.optim.SGD(optim_params, init_lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    train_name = []
    val_name = []
    train_name,val_name=gettrain_val(r'data/train_set1024/train_data')
    test_name = gettest(r'data/train_set1024/test_data')

    train_dataset = ROIDataset1(path=r'data/train_set1024/train_data', load=train_name, key=get_label,
                               mode='classification', gen_p=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = ROIDataset1(path=r'data/train_set1024/train_data', load=val_name, key=get_label, mode='classification',
                             gen_p=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = ROIDataset1(path=r'data/train_set1024/test_data', load=test_name, key=get_label,
                              mode='classification', gen_p=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1388)

    for epoch in range(1000):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device)
        # evaluate on validation set
        acc1 = validate(test_loader, model, criterion, device)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        print(best_acc1)
        # save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best)





def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output,_,_,_ = model(images,images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            progress.display(i)


def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _, _, _ = model(images, images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 5 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


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
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
