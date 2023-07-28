#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import random
import contextlib


@contextlib.contextmanager
def timer(description):
    """Recording the running time"""

    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        run_time = end_time - start_time
        return f"{description} --- Done in {run_time:.3f}s"


def time_string():
    isotimeformat = "%Y-%m-%d %X"
    string = f"[{time.strftime(isotimeformat, time.gmtime(time.time()))}]"
    return string


def time_file_str():
    isotimeformat = "%Y-%m-%d"
    string = f"{time.strftime(isotimeformat, time.gmtime(time.time()))}"
    return string + "-{}".format(random.randint(1, 10000))


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs

