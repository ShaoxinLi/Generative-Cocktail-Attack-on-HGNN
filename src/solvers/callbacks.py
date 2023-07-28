#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import copy
import torch
import shutil
import numpy as np
from ..utils.train import AverageMeter, TimeMeter
from ..utils.file import check_dir, save_record
from ..utils.time import convert_secs2time
from ..utils.plot import plot_records


class Callback(object):

    def set_state(self, state):
        self.state = state

    def on_start(self):
        pass

    def on_epoch_start(self):
        pass

    def on_loader_start(self):
        pass

    def on_batch_start(self):
        pass

    def on_batch_end(self):
        pass

    def on_loader_end(self):
        pass

    def on_epoch_end(self):
        pass

    def on_end(self):
        pass

    def on_after_backward(self):
        """Called after ``loss.backward()`` but before optimizer does anything."""
        pass


class Callbacks(Callback):
    """Class that combines multiple callbacks into one. For internal use only"""

    def __init__(self, callbacks):
        super(Callbacks, self).__init__()
        if callbacks is None:
            callbacks = [Callback()]
        self.callbacks = callbacks

    def set_state(self, state):
        for callback in self.callbacks:
            callback.set_state(state)

    def on_start(self):
        for callback in self.callbacks:
            callback.on_start()

    def on_epoch_start(self):
        for callback in self.callbacks:
            callback.on_epoch_start()

    def on_loader_start(self):
        for callback in self.callbacks:
            callback.on_loader_start()

    def on_batch_start(self):
        for callback in self.callbacks:
            callback.on_batch_start()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_loader_end(self):
        for callback in self.callbacks:
            callback.on_loader_end()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_after_backward(self):
        for callback in self.callbacks:
            callback.on_after_backward()


class MetricTracker(Callback):
    def __init__(self, metrics):
        super(MetricTracker, self).__init__()
        self.metrics = metrics      # dict of functions

    def on_start(self):
        for metric_name in self.metrics.keys():
            self.state.metric_meters[metric_name] = AverageMeter(name=metric_name)

    def on_loader_start(self):
        # reset all meters
        for meter in self.state.loss_meters.values():
            meter.reset()
        for meter in self.state.metric_meters.values():
            meter.reset()

    @torch.no_grad()
    def on_batch_end(self):
        for loss_name, loss in self.state.losses.items():
            self.state.loss_meters[loss_name].update(loss, n=self.state.num_samples_in_batch)
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(**{**self.state.logits, **self.state.labels})
            if isinstance(metric_value, (list, tuple)):     # just take top one metric
                metric_value = metric_value[0]
            self.state.metric_meters[metric_name].update(metric_value, self.state.num_samples_in_batch)

    def on_loader_end(self):
        if self.state.is_train:
            self.state.train_loss_meters = copy.deepcopy(self.state.loss_meters)
            self.state.train_metric_meters = copy.deepcopy(self.state.metric_meters)
        else:
            self.state.eval_loss_meters = copy.deepcopy(self.state.loss_meters)
            self.state.eval_metric_meters = copy.deepcopy(self.state.metric_meters)


class MetricPlotter(Callback):
    def __init__(self, exp_dir):
        super(MetricPlotter, self).__init__()
        self.exp_dir = exp_dir

    def on_start(self):
        check_dir(self.exp_dir)
        self.his_file_path = os.path.join(self.exp_dir, "train_history.csv")

    def on_end(self):
        d = {}
        for loss_name in self.state.train_loss_meters.keys():
            d.update({f"history_{loss_name}": [f"train_{loss_name}"]})
        for metric_name in self.state.train_metric_meters.keys():
            d.update({f"history_{metric_name}": [f"train_{metric_name}"]})
        if self.state.eval_loss_meters is not None:
            for loss_name in self.state.eval_loss_meters.keys():
                d[f"history_{loss_name}"].append(f"val_{loss_name}")
        if self.state.eval_metric_meters is not None:
            for metric_name in self.state.eval_metric_meters.keys():
                d[f"history_{metric_name}"].append(f"val_{metric_name}")

        for img_name, keys in d.items():
            plot_records(
                csv_file=self.his_file_path, keys=keys, img_name=img_name,
                rolling_window_size=1, average_window_size=1
            )


class FileLogger(Callback):
    def __init__(self, logger, num_epochs, verbose, print_freq=100):
        super(FileLogger, self).__init__()
        self.logger = logger
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_freq = print_freq
        self.timer = TimeMeter()

    def on_start(self):
        self.timer.reset_at_start()

    def on_epoch_start(self):
        if self.state.epoch > 0:
            need_hour, need_mins, need_secs = convert_secs2time(self.timer.epoch_time.avg * (self.num_epochs - self.state.epoch))
            self.need_time = f"[Need: {need_hour:02d}:{need_mins:02d}:{need_secs:02d}]"
        else:
            self.need_time = None

    def on_loader_start(self):
        self.timer.reset_at_loader_start()
        if self.state.is_train:
            desc = f"{'=' * 20}> Train [Epoch={self.state.epoch_log:04d}/{self.num_epochs:04d}]"
            if self.need_time is not None:
                desc += f"{self.need_time:s}"
            self.logger.info(desc)

    def on_batch_start(self):
        self.timer.clock_at_batch_start()

    def on_batch_end(self):
        self.timer.clock_at_batch_end()
        if self.verbose and self.state.is_train and (self.state.iteration + 1) % self.print_freq == 0:
            desc = f"Epoch [{self.state.epoch_log:04d}][{self.state.iteration + 1:04d}/{self.state.num_batches:04d}] "
            desc += self._format_meters(self.state.loss_meters, self.state.metric_meters)
            self.logger.info(desc)

    def on_loader_end(self):
        if self.state.is_train and self.verbose:
            d_time = self.timer.data_time.avg
            b_time = self.timer.batch_time.avg
            self.logger.info(f"Epoch [{self.state.epoch_log:04d}] Data Time: {d_time:.3f}s Batch Time: {b_time:.3f}s")

    def on_epoch_end(self):
        self.timer.clock_at_epoch_end()
        desc = f"Epoch [{self.state.epoch_log:04d}] "
        desc += f"{self._format_meters(self.state.train_loss_meters, self.state.train_metric_meters, 'Train')}"
        if self.state.eval_loss_meters is not None:
            desc += f"{self._format_meters(self.state.eval_loss_meters, self.state.eval_metric_meters, 'Val')}"
        self.logger.info(desc)

    @staticmethod
    def _format_meters(loss_meters, metric_meters, prefix=""):
        s = ""
        for loss_name, meter in loss_meters.items():
            s += f" | {prefix} {' '.join([i.capitalize() for i in loss_name.split('_')])}: {meter.avg.item():.3f}"
        for metric_name, meter in metric_meters.items():
            s += f" | {prefix} {' '.join([i.capitalize() for i in metric_name.split('_')])}: {meter.avg.item():.3f}"
        return s


class Recorder(Callback):

    def __init__(self, exp_dir):
        super(Recorder, self).__init__()
        self.exp_dir = exp_dir

    def on_start(self):
        check_dir(self.exp_dir)
        self.his_file_path = os.path.join(self.exp_dir, "train_history.csv")

    def on_epoch_end(self):
        record = {"epoch": self.state.epoch_log}
        for loss_name, meter in self.state.train_loss_meters.items():
            record.update({f"train_{loss_name}": meter.avg.item()})
        for metric_name, meter in self.state.train_metric_meters.items():
            record.update({f"train_{metric_name}": meter.avg.item()})
        if self.state.eval_loss_meters is not None:
            for loss_name, meter in self.state.eval_loss_meters.items():
                record.update({f"val_{loss_name}": meter.avg.item()})
        if self.state.eval_metric_meters is not None:
            for metric_name, meter in self.state.eval_metric_meters.items():
                record.update({f"val_{metric_name}": meter.avg.item()})
        save_record(self.his_file_path, **record)


class CheckpointSaver(Callback):

    def __init__(self, exp_dir, ckpt_name_template="checkpoint_ep{epoch:04d}_{monitor}{value:.3f}.pth.tar",
                 monitor=None, mode="min"):
        super(CheckpointSaver, self).__init__()
        self.exp_dir = exp_dir
        self.ckpt_name_template = ckpt_name_template
        self.monitor = monitor
        if mode == "min":
            self.best = torch.tensor(999999999.)
            self.monitor_op = torch.less
        elif mode == "max":
            self.best = -torch.tensor(999999999.)
            self.monitor_op = torch.greater

    def on_start(self):
        check_dir(self.exp_dir)

    def on_epoch_end(self):
        if self.monitor is not None:
            current = self.get_monitor_value()
            if self.monitor_op(current, self.best):
                epoch = self.state.epoch_log
                self.best = current
                ckpt_name = self.ckpt_name_template.format(epoch=epoch, monitor=self.monitor, value=current.item())
                ckpt_file_path = os.path.join(self.exp_dir, ckpt_name)
                self._save_checkpoint(self.state.save_dict, ckpt_file_path)
        else:
            ckpt_file_path = os.path.join(self.exp_dir, "checkpoint.pth.tar")
            self._save_checkpoint(self.state.save_dict, ckpt_file_path)

    def on_end(self):
        if not os.path.exists(os.path.join(self.exp_dir, "checkpoint.pth.tar")):
            exist_files = [f for f in os.listdir(self.exp_dir) if os.path.isfile(os.path.join(self.exp_dir, f))]
            ckpt_files = [f for f in exist_files if f.startswith("checkpoint")]
            if ckpt_files:
                epochs = np.array([int(re.findall(r"ep(\d+)", f)[0]) for f in ckpt_files])
                src_path = os.path.join(self.exp_dir, ckpt_files[int(epochs.argmax())])
                des_path = os.path.join(self.exp_dir, "checkpoint.pth.tar")
                shutil.copyfile(src_path, des_path)
            else:
                raise FileNotFoundError(f"Can't find any checkpoint files in {self.exp_dir}")

    @staticmethod
    def _save_checkpoint(save_dict, ckpt_file_path):

        # status is a dict
        for k, v in save_dict.items():
            if isinstance(v, torch.nn.Module) or isinstance(v, torch.optim.Optimizer):
                if hasattr(v, "module"):    # used for saving DDP models
                    save_dict[k] = v.module.state_dict()
                else:
                    save_dict[k] = v.state_dict()
        torch.save(save_dict, ckpt_file_path)

    def get_monitor_value(self):
        value = None
        for loss_name, meter in self.state.loss_meters.items():
            if loss_name == self.monitor:
                value = meter.avg
                break
        if value is None:
            for metric_name, metric_meter in self.state.metric_meters.items():
                if metric_name == self.monitor:
                    value = metric_meter.avg
                    break
        if value is None:
            raise ValueError(f"CheckpointSaver can't find {self.monitor} value to monitor")
        return value


class EarlyStopper(Callback):

    def __init__(self, monitor, patience, mode="min", delta=0.):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        if mode == "min":
            self.best = torch.tensor(99999.)
            self.monitor_op = torch.less
        elif mode == "max":
            self.best = -torch.tensor(99999.)
            self.monitor_op = torch.greater
        self.delta = delta

    def on_start(self):
        self._reset()

    def on_epoch_end(self):

        current = self.get_monitor_value()
        if self.mode == "min":
            _current = current + self.delta
        else:
            _current = current - self.delta
        if self.monitor_op(_current, self.best):
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter == self.patience:
                self.state.early_stop = True

    def _reset(self):
        self.counter = 0
        self.state.early_stop = False

    def get_monitor_value(self):
        value = None
        for loss_name, meter in self.state.loss_meters.items():
            if loss_name == self.monitor:
                value = meter.avg
                break
        if value is None:
            for metric_name, metric_meter in self.state.metric_meters.items():
                if metric_name == self.monitor:
                    value = metric_meter.avg
                    break
        if value is None:
            raise ValueError(f"EarlyStopper can't find {self.monitor} value to monitor")
        return value


class AttackInfoLogger(Callback):

    def __init__(self, logger, verbose):
        super(AttackInfoLogger, self).__init__()
        self.logger = logger
        self.verbose = verbose
        # because only this callback uses the following AverageMeter so we can initialize here.
        self.nonzero_ratio_meter = AverageMeter()
        self.num_pert_edges_meter = AverageMeter()

    def on_loader_start(self):
        if self.state.is_train:
            self.nonzero_ratio_meter.reset()
        else:
            self.num_pert_edges_meter.reset()

    @torch.no_grad()
    def on_batch_end(self):
        if self.state.flip_prob_dict:
            flip_probs = list(self.state.flip_prob_dict.values())
            if self.state.is_train:
                self.flip_probs = torch.cat(flip_probs)
                num_nonzeros = torch.nonzero(self.flip_probs).size(0)
                nonzero_ratio = num_nonzeros / self.flip_probs.size(0)
                self.nonzero_ratio_meter.update(nonzero_ratio * 100., n=self.state.num_samples_in_batch)
            else:
                self.flip_probs = torch.cat(flip_probs)
                self.num_pert_edges_meter.update(self.flip_probs.size(0), n=self.state.num_samples_in_batch)

    def on_loader_end(self):
        if self.verbose and self.state.is_train:
            topk_probs = torch.topk(self.flip_probs, 20)[0].tolist()
            topk_probs = [round(prob, 3) for prob in topk_probs]
            self.logger.info(f"Epoch [{self.state.epoch_log:04d}] Top-20 flip probabilities of last batch of training samples: {topk_probs}")

    def on_epoch_end(self):
        desc = f"Epoch [{self.state.epoch_log:04d}] "
        desc += f" | Training Nonzero Ratio: {self.nonzero_ratio_meter.avg:.3f}"
        desc += f" | Validating Number of Perturbed Edges: {self.num_pert_edges_meter.avg:.3f}"
        self.logger.info(desc)
