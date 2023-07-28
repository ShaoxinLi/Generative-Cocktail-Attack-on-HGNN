#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import hashlib
import pathlib


def check_dir(dir):
    """Create a directory if it doesn't exist"""

    if isinstance(dir, pathlib.PosixPath):
        if not dir.exists():
            dir.mkdir(parents=True)
    elif isinstance(dir, pathlib.WindowsPath):
        if not dir.exists():
            dir.mkdir(parents=True)
    elif isinstance(dir, str):
        if not os.path.exists(dir):
            os.makedirs(dir)
    else:
        assert False


def list_dirs(dir):
    """List all directories"""

    dirs = []
    for name in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, name)):
            dirs.append(os.path.join(dir, name))
    dirs.sort()
    return dirs


def get_exp_dir(root_dir, *args):
    """Get the result directory"""

    check_dir(root_dir)
    dir_name = ""
    for i, arg in enumerate(args):
        if i == 0:
            dir_name += str(arg)
        elif arg:
            dir_name += f"_{str(arg)}"
    exp_dir = os.path.join(root_dir, dir_name)
    return exp_dir


def save_record(csv_file, **kwargs):
    """Save records to a csv file"""

    record = {k: v for k, v in kwargs.items() if v is not None}
    file_existence = os.path.exists(csv_file)
    with open(csv_file, "a+") as f:
        writer = csv.writer(f)
        if not file_existence:
            if "epoch" in record.keys():
                other_keys = list(record.keys())
                other_keys.remove("epoch")
                header = ["epoch"] + other_keys
            else:
                header = list(record.keys())
            writer.writerow(header)
        else:
            with open(csv_file, "r") as g:
                reader = csv.reader(g)
                header = next(reader)
        row = [f"{record[i]:.3f}" if isinstance(record[i], float) else f"{record[i]}" for i in header]
        writer.writerow(row)


def compute_md5(raw_bytes):
    """Compute the md5 hash"""

    md5_obj = hashlib.md5()
    md5_obj.update(raw_bytes)
    return md5_obj.hexdigest()


