# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from functools import partial
from torch import optim as optim
import torch, math

try:
    from apex.optimizers import FusedAdam, FusedLAMB
    from apex.multi_tensor_apply import multi_tensor_applier
    import apex_C, amp_C
except:
    FusedAdam = None
    FusedLAMB = None
    print("To use FusedLAMB / FusedAdam / Slamb_V2, please install apex.")


def build_optimizer(config, model, simmim=False, is_pretrain=False):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    if simmim:
        if is_pretrain:
            parameters = get_pretrain_param_groups(model, skip, skip_keywords)
        else:
            depths = config.MODEL.SWIN.DEPTHS if config.MODEL.TYPE == 'swin' else config.MODEL.SWINV2.DEPTHS
            num_layers = sum(depths)
            get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
            scales = list(config.TRAIN.LAYER_DECAY ** i for i in reversed(range(num_layers + 2)))
            parameters = get_finetune_param_groups(model, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, get_layer_func, scales, skip, skip_keywords)
    else:
        parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'fused_adam':
        optimizer = FusedAdam(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'fused_lamb':
        optimizer = FusedLAMB(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY,
                              max_grad_norm=config.TRAIN.CLIP_GRAD)
    elif opt_lower == 'lamb':
        optimizer = Lamb(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY, debias=config.TRAIN.debias)
    elif opt_lower == 'slamb':
        optimizer = Slamb(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY,
                          compress_ratio=config.TRAIN.SLAMB.compress_ratio,
                          beta3=config.TRAIN.SLAMB.beta3,
                          local_steps=config.TRAIN.SLAMB.local_steps,
                          c_max=config.TRAIN.SLAMB.c_max,
                          c_min=config.TRAIN.SLAMB.c_min,
                          grad_size_thr=config.TRAIN.SLAMB.grad_size_thr,
                          debias=config.TRAIN.debias
                          )
    elif opt_lower == 'slamb_v2':
        optimizer = Slamb_V2(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY,
                          compress_ratio=config.TRAIN.SLAMB.compress_ratio,
                          beta3=config.TRAIN.SLAMB.beta3,
                          local_steps=config.TRAIN.SLAMB.local_steps,
                          c_max=config.TRAIN.SLAMB.c_max,
                          c_min=config.TRAIN.SLAMB.c_min,
                          grad_size_thr=config.TRAIN.SLAMB.grad_size_thr,
                          debias=config.TRAIN.debias
                          )

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


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


class Slamb_V2(torch.optim.Optimizer):
    r"""Implements Slamb algorithm with fused operations.
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
    """    r"""Implements Slamb algorithm.
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
        super(Slamb_V2, self).__init__(params, defaults)

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
        self.world_size = 1
        self.grad_size_thr = grad_size_thr
        self.mask_filter = None
        self.flat_mask_counter = None
        self.flat_m = None
        self.flat_v = None
        self.freshness = None
        self.size_per_param = []

        if multi_tensor_applier.available:
            import amp_C
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
        else:
            raise RuntimeError('apex.contrib.optimizers.FusedLAMB requires cuda extensions')

    def sync_moment(self):
        if torch.distributed.get_world_size() > 1:
            self.world_size = torch.distributed.get_world_size()

        flat_sub = torch.masked_select(self.flat_m, self.freshness == 1)
        torch.distributed.all_reduce(flat_sub)
        flat_sub.data.div_(self.world_size)
        self.flat_m[self.freshness == 1] = flat_sub

    def sync_params(self, p_all):
        if torch.distributed.get_world_size() > 1:
            self.world_size = torch.distributed.get_world_size()

        flat_param = torch.cat([p.flatten() for p in p_all])
        torch.distributed.all_reduce(flat_param)
        flat_param.data.div_(self.world_size)

        new_p_all = torch.split(flat_param, self.size_per_param)
        for p, new_p in zip(p_all, new_p_all):
            p.data = new_p.data.view(p.shape)

    def generate_mask(self, p_all):
        if self.freshness is None:
            self.freshness = torch.ones(self.flat_grad_size, dtype=self.grad_dtype, device='cuda')

        if self.compression == 'randomk':
            torch.manual_seed(self.global_step)
            flat_mask = torch.cuda.FloatTensor(self.flat_grad_size).uniform_() < self.compress_ratio

            # always sync low-dim grads
            i = 0
            for p in p_all:
                if p.dim() == 1 or p.numel() < self.grad_size_thr:
                    flat_mask[i:i + p.numel()] = True
                i += p.numel()

            self.freshness.mul_(self.beta3)
            self.freshness[flat_mask] = 1.0  # todo

            del flat_mask

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

        # create separate grad lists for fp32 and fp16 params
        g_all_32, g_all_16 = [], []
        p_all, g_all = [], []

        self.size_per_param = []
        self.grad_dtype = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p_all.append(p)
                g_all.append(p.grad.data)
                self.size_per_param.append(p.numel())
                self.grad_dtype = p.grad.dtype

                if p.dtype == torch.float32:
                    g_all_32.append(p.grad.data)
                elif p.dtype == torch.float16:
                    g_all_16.append(p.grad.data)
                else:
                    raise RuntimeError('FusedLAMB only support fp16 and fp32.')
        self.flat_grad_size = sum([p.numel() for p in p_all])

        if self.grad_pre_normalization:
            g_norm_32, g_norm_16 = 0.0, 0.0
            # compute grad norm for two lists
            if len(g_all_32) > 0:
                g_norm_32 = multi_tensor_applier(self.multi_tensor_l2norm,
                                                 self._dummy_overflow_buf,
                                                 [g_all_32], False)[0].item()
            if len(g_all_16) > 0:
                g_norm_16 = multi_tensor_applier(self.multi_tensor_l2norm,
                                                 self._dummy_overflow_buf,
                                                 [g_all_16], False)[0].item()

            # blend two grad norms to get global grad norm
            global_grad_norm = math.sqrt(g_norm_32 * g_norm_32 + g_norm_16 * g_norm_16)
            self.global_grad_norm = max(global_grad_norm, self.max_grad_norm)

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            self.global_step = group['step']

            for p in group['params']:
                if p.grad is None:
                    continue

                if p.grad.data.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                if self.grad_pre_normalization:
                    p.grad.data.div_(self.global_grad_norm)
                state = self.state[p]
                state['weight_decay'] = group['weight_decay']
                # Perform decoupled weight decay
                # if group['weight_decay'] != 0:
                #     p.data.mul_(1 - group['lr'] * group['weight_decay'])

        if self.flat_m is None:
            self.flat_m = torch.zeros(self.flat_grad_size, dtype=torch.float32, device='cuda')
            self.flat_v = torch.zeros(self.flat_grad_size, dtype=torch.float32, device='cuda')

        beta1, beta2 = self.param_groups[0]['betas']
        eps = self.param_groups[0]['eps']
        lr = self.param_groups[0]['lr']

        flat_grad = torch.cat([g.flatten() for g in g_all])
        self.flat_m.mul_(beta1).add_(flat_grad, alpha=1 - beta1)
        self.flat_v.mul_(beta2).addcmul_(flat_grad, flat_grad, value=1 - beta2)
        del flat_grad

        self.generate_mask(p_all)
        self.sync_moment()

        bias_correction1 = 1 - beta1 ** self.global_step
        bias_correction2 = 1 - beta2 ** self.global_step

        flat_adam_step = (self.flat_m / self.flat_v.sqrt().add(eps))
        flat_adam_step.mul_(bias_correction2 ** 0.5 / bias_correction1)
        flat_adam_step *= self.freshness == 1
        tensor_inlist = torch.split(flat_adam_step, self.size_per_param)
        _, max_u = multi_tensor_applier(self.multi_tensor_l2norm, self._dummy_overflow_buf, [tensor_inlist], True)
        del flat_adam_step, tensor_inlist

        flat_adam_step = (self.flat_m / self.flat_v.sqrt().add(eps))
        flat_adam_step.mul_(bias_correction2 ** 0.5 / bias_correction1)
        flat_adam_step *= self.freshness < 1
        tensor_inlist = torch.split(flat_adam_step, self.size_per_param)
        _, min_u = multi_tensor_applier(self.multi_tensor_l2norm, self._dummy_overflow_buf, [tensor_inlist], True)
        del flat_adam_step, tensor_inlist

        flat_param = torch.cat([p.flatten() for p in p_all])
        flat_param *= self.freshness == 1
        tensor_inlist = torch.split(flat_param, self.size_per_param)
        _, max_p = multi_tensor_applier(self.multi_tensor_l2norm, self._dummy_overflow_buf, [tensor_inlist], True)
        del flat_param, tensor_inlist

        flat_param = torch.cat([p.flatten() for p in p_all])
        flat_param *= self.freshness < 1
        tensor_inlist = torch.split(flat_param, self.size_per_param)
        _, min_p = multi_tensor_applier(self.multi_tensor_l2norm, self._dummy_overflow_buf, [tensor_inlist], True)
        del flat_param, tensor_inlist

        max_phi_all = torch.where(
                        (max_p > 0) * (max_u > 0),
                        max_p / max_u,
                        torch.ones_like(max_p)
                    )
        min_phi_all = torch.where(
                        (min_p > 0) * (min_u > 0),
                        min_p / min_u,
                        torch.zeros_like(min_p)
                    )

        max_phi_all = torch.clamp(max_phi_all, self.c_min, self.c_max)
        min_phi_all = torch.clamp(min_phi_all, 0, self.c_max)

        flat_adam_step = (self.flat_m / self.flat_v.sqrt().add(eps))
        flat_adam_step.mul_(bias_correction2 ** 0.5 / bias_correction1)
        u_all = torch.split(flat_adam_step, self.size_per_param)
        r_all = torch.split(self.freshness, self.size_per_param)

        for p, u, max_phi, min_phi, r in zip(p_all, u_all, list(max_phi_all), list(min_phi_all), r_all):
            u = u.view(p.shape)
            r = r.view(p.shape)
            lr_min = lr / math.sqrt(self.world_size)
            state = self.state[p]
            if state['weight_decay'] != 0:
                u.data.add_(p.data, alpha=state['weight_decay'])
            p.data.add_(- lr * max_phi * u * r - lr_min * min_phi * u * (1 - r))

        if self.global_step % self.local_steps == 1:
            self.sync_params(p_all)
        return loss