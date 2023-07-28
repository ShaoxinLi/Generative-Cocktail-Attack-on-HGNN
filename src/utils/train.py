#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch
import random
from ..utils.file import check_dir


def seed_everything(seed):

    assert seed is not None, f"Please set seeds before proceeding"
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = True


class CheckpointIO(object):

    def __init__(self, ckpt_dir, device, **kwargs):
        check_dir(ckpt_dir)
        self.device = device
        self.ckpt_file_path = os.path.join(ckpt_dir, "checkpoint.pth.tar")
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self):
        for k, v in self.module_dict.items():
            if isinstance(v, torch.nn.Module) or isinstance(v, torch.optim.Optimizer):
                if hasattr(v, "module"):
                    self.module_dict[k] = v.module.state_dict()
                else:
                    self.module_dict[k] = v.state_dict()
        torch.save(self.module_dict, self.ckpt_file_path)

    def load(self):
        return self.load_from_path(self.ckpt_file_path)

    def load_from_path(self, ckpt_file_path):
        assert os.path.exists(ckpt_file_path), f"{ckpt_file_path} does not exist!"
        module_dict = torch.load(ckpt_file_path, map_location=self.device)
        for k, v in self.module_dict.items():
            if isinstance(v, torch.nn.Module) or isinstance(v, torch.optim.Optimizer):
                if hasattr(v, "module"):
                    v.module.load_state_dict(module_dict[k])
                else:
                    v.load_state_dict(module_dict[k])
            else:
                self.module_dict[k] = module_dict[k]
        return self.module_dict


class AverageMeter(object):

    def __init__(self, name="Meter"):

        self.name = name
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

    def __call__(self, val):
        return self.update(val)

    def __repr__(self):
        return f"AverageMeter(name={self.name}, avg={self.avg:.3f}, count={self.count})"


class TimeMeter(object):

    def reset_at_loader_start(self):
        self.data_time = AverageMeter()
        self.batch_time = AverageMeter()
        self.batch_start = time.time()

    def reset_at_start(self):
        self.epoch_time = AverageMeter()
        self.epoch_start = time.time()

    def clock_at_batch_start(self):
        self.data_time.update(time.time() - self.batch_start, n=1)

    def clock_at_batch_end(self):
        self.batch_time.update(time.time() - self.batch_start, n=1)
        self.batch_start = time.time()

    def clock_at_epoch_end(self):
        self.epoch_time.update(time.time() - self.epoch_start, n=1)
        self.epoch_start = time.time()