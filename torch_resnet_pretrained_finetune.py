# Date: 3/8/20

import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='../dataset/split',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names, 
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=False,
                    help='use pre-trained model')
parser.add_argument('--finetune', dest='finetune', action='store_true', default=False,
                    help='fine tune pre-trained model')
parser.add_argument('--device', dest='device', default='cuda',
                    help='use cpu or cuda')
parser.add_argument('--best_model', dest='best_model', default='model_best',
                    help='name of the best model')
# parser.add_argument('--gpu', type=int, default=[5,6,7], nargs='+', help='used gpu id')
parser.add_argument('--combine', dest='combine', nargs='+', 
                    help='Average weights specified by the checkpoints')
best_acc = 0
num_classes = None

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            if '18' in arch:
                dim = 512
            elif '50' in arch:
                dim = 2048
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(dim, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = True


    def forward(self, x):
        f = self.features(x)

        if self.modelName == 'alexnet' :
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet' :
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

def main():
    global args, best_acc, num_classes
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    # TODO: change here to modify train dataset;
    traindir = os.path.join(args.data, 'train200_2')
    valdir = os.path.join(args.data, 'test')
    # Get number of classes from train directory
    num_classes = len([name for name in os.listdir(traindir)])
    print("num_classes = '{}'".format(num_classes))
    
    # create model
    if args.finetune:
        print("=> using pre-trained model '{}'".format(args.arch))
        original_model = models.__dict__[args.arch](pretrained=True)
        model = FineTuneModel(original_model, args.arch, num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.to(args.device)
    else:
        model = torch.nn.DataParallel(model).to(args.device)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.combine:
        
        weights = {}
        
        for cp in args.combine:
            print("=> loading checkpoint '{}'".format(cp))
            checkpoint = torch.load(cp, map_location=torch.device(args.device))
            state_dict = checkpoint['state_dict']
            keys = state_dict.keys() #[key for key in state_dict.keys() if key.startswith('module.classifier')]
            for key in keys:
                if key in weights.keys():
                    weights[key] = weights[key] + state_dict[key]
                else:
                    weights[key] = state_dict[key]

        for key in keys:
            state_dict[key] = weights[key] / len(args.combine)

        model.load_state_dict(state_dict)


        
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    
    test_dataset = ImageFolderWithPaths(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    print(test_dataset.class_to_idx)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        start = time.time()
        accuracy, cm = validate(val_loader, model, criterion)
        end = time.time()
        
        fp, fn, tp, tn = cm
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (tn + fp)
        fnr = fn / (tp + fn)
        f1 = 2. / (1./precision + 1./recall)
        
        print("-------- Report --------")
        print("=> average accuracy {}".format(accuracy))
        print("=> average precision {}".format(precision.mean()))
        print("=> average recall {}".format(recall.mean()))
        print("=> average fpr {}".format(fpr.mean()))
        print("=> average fnr {}".format(fnr.mean()))
        print("=> average f1 {}".format(f1.mean()))
        print("=> average validation time {}".format((end-start)/len(test_dataset)))
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc, _ = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
        }, is_best, best_model_name = args.best_model)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(args.device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        _, predicted = torch.max(output.data, 1)
        acc = (predicted == target).sum().item()
        
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        accs.update(acc / input.size(0), input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc=accs))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    tns = AverageMeter()
    fps = AverageMeter()
    fns = AverageMeter()
    tps = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    f1s = AverageMeter()
    
    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    for i, (input, target, path) in enumerate(val_loader):
        target = target.to(args.device)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        _, predicted = torch.max(output.data, 1)
        acc = (predicted == target).sum().item()
        
        
        for j in range(len(target)):
            if target[j].item() != predicted[j].item():
                print("Target: {} Predicted: {} Image: {}".format(target[j].item(), predicted[j].item(), path[j]))

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        accs.update(acc / input.size(0), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        cm = confusion_matrix(target.cpu().numpy(), predicted.cpu().numpy(), labels=[*range(6)])
        
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - fp - fn - tp
                
        fps.update(fp, input.size(0))
        fns.update(fn, input.size(0))
        tps.update(tp, input.size(0))
        tns.update(tn, input.size(0))


        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, acc=accs))

    return accs.avg, (fps.sum, fns.sum, tps.sum, tns.sum)


def save_checkpoint(state, is_best, filename='models/checkpoint.pth.tar', best_model_name='model_best'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'models/{}.pth.tar'.format(best_model_name))

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
