import argparse
import os
import random
import shutil
import time
import warnings

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
import models
import time

model_names = sorted(name for name in models.__dict__)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_ACD',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50_SVPN)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_mode', '--learning-rate-mode', default='LRnorm', type=str,
                    help='choose the lr mode you want to train this model,include LRnorm,LRfast,LRadju')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--lamda_start', default=None, type=float,
                    help='weighting value for mse loss. ')
parser.add_argument('--lamda_end', default=None, type=float,
                    help='weighting value for mse loss. ')
parser.add_argument('--epochs_lamda', default=None, type=int,
                    help='weighting value for mse loss. ')
parser.add_argument('--lamda_normA_start', default=None, type=float,
                    help='weighting value for mse loss. ')
parser.add_argument('--lamda_normA_end', default=None, type=float,
                    help='weighting value for mse loss. ')
parser.add_argument('--batches_per_epoch', default=None, type=int,
                    help='weighting value for mse loss. ')
parser.add_argument('--epochs_lamda_normA', default=None, type=int,
                    help='weighting value for mse loss. ')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.arch.startswith('resnet'):
        args.lr = 0.1
        args.batch_size=256
        args.weight_decay = 1e-4
        if args.lr_mode == 'LRnorm':
            args.epochs = 115
        elif args.lr_mode == 'LRfast':
            args.epochs = 30
        elif args.lr_mode == 'LRadju':
            args.epochs = 50
    elif args.arch.startswith('mobilenet'):
        args.weight_decay = 4e-5
        if args.lr_mode == 'LRnorm':
            args.lr = 0.045
            args.batch_size = 96
        else:
            args.lr = 0.06
            if args.lr_mode == 'LRfast':
                args.batch_size = 96
                args.epochs = 100
            elif args.lr_mode == 'LRadju':
                args.batch_size = 192
                args.epochs = 150
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.local_rank == -1:
            args.local_rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.local_rank = args.local_rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.local_rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_mse = nn.MSELoss().cuda(args.gpu)
    
    # for param in model.module.fc.parameters():
    #     param.requires_grad = False
    # for param in model.module.layer_reduce.parameters():
    #     param.requires_grad = True
    # for param in model.module.layer_reduce_bn.parameters():
    #     param.requires_grad = True
    
    # for name, param in model.named_parameters(): 
    #     if param.requires_grad:
    #         print(name) 

    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # if args.evaluate:
    if False:
        print('lala')
        valObj, prec1, prec5 = validate(val_loader, model, criterion, args)
        print(prec1)
        return
    
    lamda_normA_rator = Adjust_lamda_normA_rate(args)
    lamda_rator = Adjust_lamda_rate(args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        trainObj, trainObj_mse, trainObj_cls, trainObj_normA, top1, top5 = train(train_loader, model, criterion, criterion_mse, optimizer, epoch, args, lamda_normA_rator,lamda_rator)

        # evaluate on validation set
        valObj, prec1, prec5 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = prec1 > best_acc1
        best_acc1 = max(prec1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.local_rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        
        save_flg = ""
        if is_best:
            save_flg='model saved'

        print('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.4f}\t{:.2f}%\t'.format(epoch, trainObj, trainObj_mse, trainObj_cls, trainObj_normA, top1,valObj, prec1)+time.strftime('%H:%M:%S\t',time.localtime(time.time()))+save_flg)


def train(train_loader, model, criterion, criterion_mse, optimizer, epoch, args,lamda_normA_rator,lamda_rator):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_cls = AverageMeter('Loss', ':.4e')
    losses_mse = AverageMeter('Loss', ':.4e')
    losses_mse_normA = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        # compute output
        output, x_cov = model(images)
        
        # with torch.no_grad():
        batch, d, d= x_cov.size()
        dtype = x_cov.dtype
        I = torch.eye(d,d,device = x_cov.device).view(1, d, d).repeat(batch,1,1).type(dtype)
        normA = x_cov.mul(I).sum(dim=1).sum(dim=1)
        x_cov = x_cov.div(normA.view(batch,1,1).expand_as(x_cov)+1e-5)
        
        with torch.no_grad():
            loss_mse_norm = torch.mean(normA)
            loss_mse_toprint = torch.mean(torch.sum(x_cov**2,dim=(1,2)))
        loss_mse = torch.mean((torch.sum(x_cov**2,dim=(1,2))-0.0530)**2)
        loss_mse_normA = torch.mean((normA-215.0)**2) #174 
        

        # loss_mse = torch.mean(torch.sum(x_cov**2,dim=(1,2)))
        loss_cls = criterion(output, target)
        # print(loss_mse_normA)
        # exit()
        
        if epoch<=108:
            lamda_normA = lamda_normA_rator.current_lamda_normA(False)
            lamda = lamda_rator.current_lamda(False)
        else:
            lamda_normA = lamda_normA_rator.current_lamda_normA(True)
            lamda = lamda_rator.current_lamda(True)
        
        
        loss = lamda*loss_mse + loss_cls+lamda_normA*loss_mse_normA
        # loss = loss_cls
        # # loss = loss_mse
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        losses.update(loss.item(), images.size(0))
        losses_mse.update(loss_mse_toprint.item(), images.size(0))
        losses_cls.update(loss_cls.item(), images.size(0))
        losses_mse_normA.update(loss_mse_norm.item(), images.size(0))
        
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # print(losses_mse_normA.avg)
        # print(losses_mse.avg)
        # print(losses_cls.avg)
        # print(top1.avg)
    
    print(lamda_normA)
    print(lamda)

    return losses.avg, losses_mse.avg, losses_cls.avg, losses_mse_normA.avg, top1.avg, top5.avg
    

        

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')


    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            # compute output
            output, _ = model(images)

            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    # if state["epoch"] % 10 == 0:
    #     torch.save(state, 'checkpoint{}.pth.tar'.format(state["epoch"]))   
    if is_best:
        shutil.copyfile(filename, 'best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, args):
    if args.arch.startswith('resnet'):
        if args.lr_mode == 'LRnorm':
            if epoch<35:
                lr = args.lr
            elif (epoch>=35) and (epoch<60):
                lr = args.lr * 0.1
            elif (epoch>=60) and (epoch<90):
                lr= args.lr*0.01
            elif (epoch>=90) and (epoch<100):
                lr= args.lr*0.001
            elif (epoch>=100) and (epoch<109):
                lr = args.lr*0.01
            else:
                lr = args.lr*0.001
                
                            
        elif args.lr_mode == 'LRfast':
            e_start = 1
            e_end = 53
            p = 11.0
            lr = args.lr * (1-(epoch-e_start)/(e_end-e_start)) ** p
        elif args.lr_mode == 'LRadju':
            e_start = 1
            e_end = 50
            p = 2.0
            lr = args.lr * (1-(epoch-e_start)/(e_end-e_start)) ** p
    elif args.arch.startswith('mobilenet'):
        if args.lr_mode == 'LRnorm':
            lr = args.lr *(0.98)**epoch
        elif args.lr_mode == 'LRfast':
            lr = args.lr *(0.92)**epoch
        elif args.lr_mode == 'LRadju':
            if epoch <=50:
                lr = args.lr - (args.lr - 0.01)/50*(epoch)
            elif epoch > 50 and epoch <= 100:
                lr = 0.01 - (0.01-0.001)/50*(epoch -50)
            elif epoch > 100 and epoch <=150:
                lr = 0.001 - (0.001-0.00001)/50*(epoch-100)
            else :
                lr = 0.00001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Adjust_lamda_normA_rate(object):
    def __init__(self, args):
        self.lamda_normA_start = args.lamda_normA_start
        self.lamda_normA_end = args.lamda_normA_end
        self.iters = args.batches_per_epoch*args.epochs_lamda_normA
        self.counter = -1
    
    def current_lamda_normA(self,is_step2):
        if not is_step2:
            if self.counter <= self.iters:
                self.counter += 1
            return self.lamda_normA_start+ (self.lamda_normA_end-self.lamda_normA_start)/self.iters*self.counter
        else:
            return self.lamda_normA_end

class Adjust_lamda_rate(object):
    def __init__(self, args):

        self.lamda_start = args.lamda_start
        self.lamda_end = args.lamda_end
        self.iters = args.batches_per_epoch*args.epochs_lamda
        self.counter = -1
    
    def current_lamda(self,is_step2):
        if not is_step2:
            if self.counter <= self.iters:
                self.counter += 1
            return self.lamda_start+ (self.lamda_end-self.lamda_start)/self.iters*self.counter
        else:
            return self.lamda_end
    

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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
