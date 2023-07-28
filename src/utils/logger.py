#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import logging.config
from .file import check_dir


def get_logger(log_dir, logger_name=None, log_file_name=None):
    """Initialize a logger"""

    check_dir(log_dir)
    formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")

    logger_name = "logger" if logger_name is None else logger_name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Console handler
    handler_console = logging.StreamHandler()
    handler_console.setLevel(logging.INFO)
    handler_console.setFormatter(formatter)
    logger.addHandler(handler_console)

    # File handler
    now = datetime.datetime.now()
    datatime_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_file_name = f"log-{datatime_string}.txt" if log_file_name is None else log_file_name
    log_file_path = os.path.join(log_dir, log_file_name)
    handler_file = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        maxBytes=10485760,
        backupCount=5,
        encoding="utf-8"
    )
    handler_file.setLevel(logging.INFO)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    return logger


def print_log(print_string, log):
    print(f"{print_string}")
    log.write(f"{print_string}\n")
    log.flush()