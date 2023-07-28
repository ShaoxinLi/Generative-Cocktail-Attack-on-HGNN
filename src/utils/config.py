#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import random

import torch
import torch.backends.cudnn as cudnn

from .logger import get_logger
from .file import check_dir, get_exp_dir, list_dirs


def setup_cfg(args):

    # set the result dir
    if hasattr(args, "target_net_arch"):
        exp_dir = get_exp_dir(
            args.exp_root_dir, args.exp_type, args.dataset, args.target_net_arch, args.dir_suffix
        )
    else:
        exp_dir = get_exp_dir(
            args.exp_root_dir, args.exp_type, args.dataset, args.net_arch, args.dir_suffix
        )
    exist_dirs = list_dirs(args.exp_root_dir)
    if exp_dir in exist_dirs:
        args.exp_dir = exp_dir
    else:
        exp_dirs_with_same_prefix = [i for i in exist_dirs if i.startswith(exp_dir)]
        if exp_dirs_with_same_prefix:
            exp_idx = [i.split("_")[-1] for i in exp_dirs_with_same_prefix]
            exp_idx = [int(re.findall(r"v(\d+)", i)[0]) for i in exp_idx]
            args.exp_dir = exp_dir + f"_v{max(exp_idx) + 1}"
        else:
            args.exp_dir = exp_dir + f"_v1"
    check_dir(args.exp_dir)

    # set the logger
    args.logger = get_logger(log_dir=args.exp_dir, log_file_name=None)

    # set the number of workers
    if os.name == "nt" and args.num_workers != 0:
        print("Warning: reset n_workers = 0, because running on a Windows system.")
        args.num_workers = 0

    # set the random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)

    # set the device
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)


def print_cfg(args):

    # log all contexts and hyper-parameters
    args.logger.info(f"===========================> System info:")
    python_version = sys.version.replace('\n', ' ')
    args.logger.info(f"Python version: {python_version}")
    args.logger.info(f"Torch version: {torch.__version__}")
    # args.logger.info(f"Cudnn version: {torch.backends.cudnn.version()}")
    args.logger.info(f"===========================> Hyperparameters:",)
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in sorted(state.items(), key=lambda x: x[0]):
        args.logger.info(f"{key}: {value}")

