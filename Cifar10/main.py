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
import resnet

import horovod.torch as hvd
# import wandb
# import socket
import math
import random
import numpy as np

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--data-dir', default='./data', type=str, help='dataset path')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=5, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--save-model', dest='save_model', action='store_true',
                    help='save model after training')

parser.add_argument('--wandb', action='store_true',  default=False,
                    help='use wandb to log training metrics')
parser.add_argument("--local_rank",
                    type=int,
                    default=os.getenv('LOCAL_RANK', -1),
                    help="local_rank for distributed training on gpus")
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
parser.add_argument('--optimizer', default='SGD',
                    help='which optimizer to use: SGD, ADAM, LAMB, SLAMB')
parser.add_argument('--beta3', default=0.95, type=float,
                    help='beta3 for SLAMB')
parser.add_argument('--compress_ratio', default=0.1, type=float,
                    help='compress_ratio for SLAMB')
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")


best_prec1 = 0


def setup_training(args):

    # args.comm_backend = 'nccl'
    # args.global_step = 0
    # args.optimizer_profiling = False
    args.lr_warmup = False

    assert (torch.cuda.is_available())
    hvd.init()
    args.rank = hvd.rank()
    args.world_size = hvd.size()
    args.local_rank = hvd.local_rank()
    print("env args.rank", args.rank)
    print("env args.local_rank", args.local_rank)
    print("env world_size", args.world_size)

    args.n_gpus_per_node = torch.cuda.device_count()
    args.n_nodes = int(args.world_size / torch.cuda.device_count())

    os.environ['MASTER_ADDR'] = args.dist_url
    os.environ['MASTER_PORT'] = '12345'
    # some dummy params to get ddp work
    os.environ['nproc_per_node'] = '1'
    os.environ['nnodes'] = '1'
    os.environ['node_rank'] = '0'

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                         world_size=args.world_size, rank=args.rank)

    torch.distributed.barrier()
    # args.arch = 'resnet110'
    model = resnet.__dict__[args.arch]()
    # model = torch.nn.DataParallel(model)
    model.cuda()

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'ADAM':
        optimizer = Lamb(model.parameters(), adam=True, lr=args.lr, weight_decay=args.weight_decay)

    elif args.optimizer == 'LAMB':
        optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.optimizer == 'SLAMB':
        optimizer = Slamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                          compress_ratio=args.compress_ratio,
                          beta3=args.beta3,
                          local_steps=50,)

    return optimizer, model, device, args


def main():
    global args, best_prec1
    args = parser.parse_args()
    optimizer, model, device, args = setup_training(args)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset_train = datasets.CIFAR10(root=args.data_dir, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    dataset_val = datasets.CIFAR10(root=args.data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    sampler_train = torch.utils.data.distributed.DistributedSampler(
        dataset_train, num_replicas=hvd.size(), rank=hvd.rank())
    # sampler_val = torch.utils.data.distributed.DistributedSampler(
    #     dataset_val, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size, shuffle=False, sampler=sampler_train,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], gamma=0.1, last_epoch=args.start_epoch - 1)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.arch in ['resnet1202', 'resnet110'] and args.lr_warmup:
            # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
            # then switch back. In this setup it will correspond for first epoch.
            if epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * 0.1
            elif epoch == 1:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

        # compression warmup
        ratios = [0.5, 0.5, 0.2, 0.1, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.008, 0.008, 0.006, 0.006, 0.003, 0.003, 0.001]
        if epoch < len(ratios):
            compress_ratio = ratios[epoch]
        else:
            compress_ratio = ratios[-1]
        compress_ratio = max(compress_ratio, args.compress_ratio)
        optimizer.compress_ratio = compress_ratio

        args.current_epoch = epoch
        # train for one epoch
        if args.rank == 0:
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']),)
            # wandb.log({'train/lr': optimizer.param_groups[0]['lr'], 'epoch': args.current_epoch,
            #            }, commit=True)
            # if args.optimizer == 'SLAMB':
            #     wandb.log({'train/compress_ratio': compress_ratio, 'epoch': args.current_epoch,
            #                }, commit=True)
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

    torch.distributed.barrier()

    if args.rank == 0:
        print('== Best Prec@1 {:.3f}'.format(best_prec1))
        # if args.wandb:
        #     wandb.log({'best_prec1': best_prec1, }, commit=True)
        #     wandb.finish()

    if args.save_model:
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, filename=os.path.join(args.save_dir, 'model.th'))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.optimizer != 'SLAMB':
            for p in model.parameters():
                torch.distributed.all_reduce(p.grad)
                p.grad.data.div_(args.world_size)

        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            # if args.rank == 0 and args.wandb:
            #     wandb.log({'train/top-1': top1.val,
            #                'train/loss': losses.val,
            #                'epoch': args.current_epoch,
            #                }, commit=True)


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0 and args.rank == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #               i, len(val_loader), batch_time=batch_time, loss=losses,
            #               top1=top1))

    if args.rank == 0:
        print(' *Test: Epoch {epoch} Prec@1 {top1.avg:.3f}'
              .format(top1=top1, epoch=args.current_epoch))
    # if args.rank == 0 and args.wandb:
    #     wandb.log({'validate/top-1': top1.avg,
    #                'epoch': args.current_epoch,
    #                }, commit=True)

    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


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


class Lamb(torch.optim.Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`__.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-6)
        weight_decay: weight decay (L2 penalty) (default: 0)
        clamp_value: clamp weight_norm in (0,clamp_value) (default: 10)
            set to a high value to avoid it (e.g 10e3)
        adam: always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes. (default: False)
        debias: debias adam by (1 - beta**step) (default: False)
    Example:
        >>> optimizer = Lamb(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ https://arxiv.org/abs/1904.00962
    Note:
        Reference code: https://github.com/cybertronai/pytorch-lamb
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.0,
            clamp_value=10,
            adam=False,
            debias=True,
            max_grad_norm=1.0,
            grad_pre_normalization=False,

    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super(Lamb, self).__init__(params, defaults)

        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias
        self.grad_pre_normalization = grad_pre_normalization
        self.max_grad_norm = max_grad_norm
        self.global_grad_norm = None


    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        if self.grad_pre_normalization:
            global_grad_norm = 0.0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        global_grad_norm += p.grad.data.pow(2).sum()
            self.global_grad_norm = max(math.sqrt(global_grad_norm.item()), self.max_grad_norm)


        for group in self.param_groups:

            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if self.grad_pre_normalization:
                    grad = grad / self.global_grad_norm

                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Debiasing
                bias_correction1 = bias_correction2 = 1.0
                if self.debias:
                    bias_correction1 = 1 - beta1 ** group['step']
                    bias_correction2 = 1 - beta2 ** group['step']
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                step_size = group['lr']

                weight_norm = torch.norm(p.data)  # .clamp(0, self.clamp_value)

                adam_step = exp_avg_hat / exp_avg_sq_hat.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)

                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = torch.ones([1], device=adam_norm.device, dtype=adam_norm.dtype)
                else:
                    trust_ratio = weight_norm / adam_norm

                if self.adam:
                    trust_ratio = 1

                update = -trust_ratio * step_size * adam_step
                p.data.add_(update)

        return loss


class Slamb(torch.optim.Optimizer):
    r"""Implements Slamb algorithm.
    It has been proposed in `Accelerated Large Batch Training with Sparse Communication`.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        clamp_value: clamp weight_norm in (0,clamp_value) (default: 10)
            set to a high value to avoid it (e.g 10e3)
        adam: always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes. (default: False)
        debias: debias adam by (1 - beta**step) (default: False)
    Example:
        >>> optimizer = Slamb(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    Note:
        Paper: https://openreview.net/pdf?id=cMmjBH5LqW
        Reference code: https://github.com/hangxu0304/SLAMB
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.0,
            clamp_value=10,
            adam=False,
            debias=True,
            max_grad_norm=1.0,
            grad_pre_normalization=False,
            compress_ratio=0.1,
            beta3=0.99,
            local_steps=100,
            c_max=1000,
            c_min=0.01,
            grad_size_thr=9000,

    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super(Slamb, self).__init__(params, defaults)

        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias
        self.grad_pre_normalization = grad_pre_normalization
        self.max_grad_norm = max_grad_norm
        self.global_grad_norm = None

        self.flat_mask = None
        self.mask_counter = {}
        self.mapping = None
        self.flat_grad_size = None
        self.c_max = c_max
        self.c_min = c_min
        self.beta3 = beta3
        self.local_steps = local_steps
        self.compression = 'randomk'
        self.compress_ratio = compress_ratio
        self.global_step = None
        self.world_size = torch.distributed.get_world_size()
        self.grad_size_thr = grad_size_thr

    def sync_moment(self):
        moment = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                moment.append(self.state[p]['exp_avg'])

        flat_raw_ = torch.cat([m.flatten() for m in moment])
        flat_sub = torch.masked_select(flat_raw_, self.flat_mask)
        torch.distributed.all_reduce(flat_sub)
        flat_sub.data.div_(self.world_size)
        flat_raw_[self.flat_mask] = flat_sub

        split_sizes = [m.numel() for m in moment]
        for m_synced, m in zip(torch.split(flat_raw_, split_sizes), moment):
            m.data = m_synced.data.view(m.size()).clone().detach()

    def sync_gradient(self):
        params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)

        flat_raw_ = torch.cat([p.grad.data.flatten() for p in params])
        flat_sub = torch.masked_select(flat_raw_, self.flat_mask)
        torch.distributed.all_reduce(flat_sub)
        flat_sub.data.div_(self.world_size)
        flat_raw_[self.flat_mask] = flat_sub

        split_sizes = [p.numel() for p in params]
        for g_synced, p in zip(torch.split(flat_raw_, split_sizes), params):
            p.grad.data = g_synced.data.view(p.grad.size()).clone().detach()

    def sync_params(self):
        for group in self.param_groups:
            for p in group['params']:
                torch.distributed.all_reduce(p)
                p.data.div_(self.world_size)

    def generate_mask(self):
        grad_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_list.append(p.grad)
        self.flat_grad_size = sum([g.numel() for g in grad_list])

        if self.mapping is None:
            torch.manual_seed(44)  # todo
            self.mapping = torch.randperm(self.flat_grad_size, dtype=torch.int32, device='cuda')

        if self.compression == 'randomk':
            num_chunks = int(1.0 / self.compress_ratio)
            j = self.global_step % num_chunks
            rand_idx = self.mapping.chunk(num_chunks)[j]
            flat_mask = torch.zeros(self.flat_grad_size, device=grad_list[0].device, dtype=grad_list[0].dtype)
            flat_mask[rand_idx.long()] += 1.0
            flat_mask = flat_mask.bool()

        # always sync small grads
        i = 0
        for g in grad_list:
            if g.dim() == 1 or g.numel() < self.grad_size_thr:
                flat_mask[i:i + g.numel()] = True
            i += g.numel()

        self.flat_mask = flat_mask

    @staticmethod
    def compute_trust_ratio(adam_tensor, weight_tensor):
        adam_norm = torch.norm(adam_tensor)
        weight_norm = torch.norm(weight_tensor)
        if weight_norm == 0 or adam_norm == 0:
            trust_ratio = torch.ones([1], device=adam_norm.device, dtype=adam_norm.dtype)
        else:
            trust_ratio = weight_norm / adam_norm
        return trust_ratio, weight_norm, adam_norm

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        if self.grad_pre_normalization:
            if self.global_grad_norm is None:
                global_grad_norm = 0.0
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            global_grad_norm += torch.norm(p.grad.data)**2
                self.global_grad_norm = global_grad_norm.item() ** 0.5
            self.global_grad_norm = max(self.global_grad_norm, self.max_grad_norm)

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            self.global_step = group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if self.grad_pre_normalization:
                    grad = grad / self.global_grad_norm

                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        self.generate_mask()
        self.sync_moment()
        # self.sync_gradient()

        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if self.grad_pre_normalization:
                    grad = grad / self.global_grad_norm

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Debiasing
                bias_correction1 = bias_correction2 = 1.0
                if self.debias:
                    bias_correction1 = 1 - beta1 ** group['step']
                    bias_correction2 = 1 - beta2 ** group['step']
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                step_size = group['lr']

                adam_step = exp_avg_hat / exp_avg_sq_hat.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                if p not in self.mask_counter:
                    self.mask_counter[p] = torch.zeros_like(p.data)
                mask = self.flat_mask[i:i + p.numel()].view(p.data.size())
                self.mask_counter[p] = (self.mask_counter[p] + 1) * (~mask)
                i += p.numel()

                if grad.dim() == 1:
                    trust_ratio, _, _ = self.compute_trust_ratio(adam_step, p.data)
                else:
                    trust_ratio_large_batch, weight_norm_large, adam_norm_large = self.compute_trust_ratio(
                        adam_step[mask], p.data[mask])
                    trust_ratio_base, weight_norm_base, adam_norm_base = self.compute_trust_ratio(adam_step[~mask],
                                                                                             p.data[~mask])
                    synced_ratio = (self.beta3 ** self.mask_counter[p])
                    step_size = synced_ratio * group['lr'] + (1 - synced_ratio) * group['lr'] / math.sqrt(self.world_size)

                    trust_ratio_min = trust_ratio_base
                    trust_ratio_max = trust_ratio_large_batch
                    trust_ratio = synced_ratio * trust_ratio_max + (1 - synced_ratio) * trust_ratio_min
                    trust_ratio = torch.clamp(trust_ratio, self.c_min, self.c_max)

                if self.adam:
                    trust_ratio = 1

                update = -trust_ratio * step_size * adam_step
                p.data.add_(update)

        self.global_grad_norm = None
        self.flat_mask = None
        if self.global_step % self.local_steps == 1:
            self.sync_params()
        return loss


if __name__ == '__main__':
    main()