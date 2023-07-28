#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import torch
import torch.nn.functional as F
from collections import defaultdict
from .callbacks import Callbacks
from ..utils import seed_everything, AverageMeter, CheckpointIO


class NodeClassifier(object):
    def __init__(self, opt_alg, lr, num_epochs, device, seed=None):
        super(NodeClassifier, self).__init__()
        self.opt_alg = opt_alg
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        self.seed = seed

    def fit(self, net, g, full_batch=True, train_loader=None, val_loader=None, callbacks=None):

        # prepare routines
        self.net = net
        self.g = g
        if "h" in self.g.ndata.keys():
            self.feat_dict = self.g.ndata["h"]
            params = self.net.parameters()
        else:
            from ..models.rgcn import RelGraphEmbed
            ntype2num = {ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes}
            if "feat" in g.ndata.keys():  # dblp
                feat_dict = RelGraphEmbed(ntype2num, 334)
            else:
                feat_dict = RelGraphEmbed(ntype2num, self.net.hidden_dim)
            self.feat_dict = feat_dict.to(self.device, non_blocking=True)
            params = itertools.chain(self.net.parameters(), self.feat_dict.parameters())
        self.optimizer, self.scheduler = self._configure_optimizers(params=params, opt_alg=self.opt_alg, lr=self.lr)
        if "edge_weight" in self.g.edata.keys():
            self.edge_weight_dict = self.g.edata["edge_weight"]
        else:
            self.edge_weight_dict = None
        self.criterion = self._set_criterion(self.g.num_classes, self.device)
        seed_everything(self.seed)
        self.callbacks = Callbacks(callbacks)
        self.state = RunnerState()
        self.callbacks.set_state(self.state)

        if full_batch:
            self.train_full_batch()
        else:
            assert train_loader is not None
            self.train_stochastic(train_loader, val_loader)

    def train_full_batch(self):
        self.callbacks.on_start()
        for epoch in range(self.num_epochs):
            self.state.epoch = epoch
            self.callbacks.on_epoch_start()

            # train one epoch
            self.net.train()
            if "h" not in self.g.ndata.keys():
                self.feat_dict.train()
            self.state.is_train = True
            self.state.num_batches = 1
            self.callbacks.on_loader_start()
            self.state.num_samples_in_batch = self.g.train_idxs.size(0)
            self.state.iteration = 0
            self.callbacks.on_batch_start()

            labels = self.g.labels[self.g.train_idxs.long()]
            logits = self.forward(self.g, None, self.feat_dict, self.edge_weight_dict)[0][self.g.train_idxs.long()]
            if self.g.num_classes == 2 and labels.dim() == 1:
                loss = self.criterion(logits, F.one_hot(labels.long(), num_classes=2).float())
            else:
                loss = self.criterion(logits, labels.long())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.callbacks.on_after_backward()
            self.optimizer.step()

            self.state.losses = {"loss": loss.detach()}
            self.state.logits = {"logits": logits.detach()}
            self.state.labels = {"labels": labels}

            self.callbacks.on_batch_end()
            if self.scheduler is not None:
                self.scheduler.step()
            self.callbacks.on_loader_end()

            # val one epoch
            with torch.no_grad():
                self.net.eval()
                if "h" not in self.g.ndata.keys():
                    self.feat_dict.eval()
                self.state.is_train = False
                self.state.num_batches = 1
                self.callbacks.on_loader_start()
                self.state.num_samples_in_batch = self.g.val_idxs.size(0)
                self.state.iteration = 0
                self.callbacks.on_batch_start()

                labels = self.g.labels[self.g.val_idxs.long()]
                logits = self.forward(self.g, None, self.feat_dict, self.edge_weight_dict)[0][self.g.val_idxs.long()]
                if self.g.num_classes == 2 and labels.dim() == 1:
                    loss = self.criterion(logits, F.one_hot(labels.long(), num_classes=2).float())
                else:
                    loss = self.criterion(logits, labels.long())

                self.state.losses = {"loss": loss.detach()}
                self.state.logits = {"logits": logits.detach()}
                self.state.labels = {"labels": labels}

                self.callbacks.on_batch_end()
                self.callbacks.on_loader_end()

            if "h" in self.g.ndata.keys():
                self.state.save_dict = {"net_state": self.net, "optimizer_state": self.optimizer}
            else:
                if "feat" in self.g.ndata.keys():  # dblp
                    self.feat_dict.embeds["author"] = self.g.ndata["feat"]["author"]
                self.state.save_dict = {"net_state": self.net, "embed_state": self.feat_dict, "optimizer_state": self.optimizer}
            self.callbacks.on_epoch_end()
            if self.state.early_stop:
                break

        self.callbacks.on_end()

    def train_stochastic(self, train_loader, val_loader):

        # start training
        self.callbacks.on_start()
        for epoch in range(self.num_epochs):
            self.state.epoch = epoch
            self.callbacks.on_epoch_start()

            # train one epoch
            self._train_epoch(train_loader)

            # val one epoch
            if val_loader is not None:
                self._eval_epoch(val_loader)

            if "h" in self.g.ndata.keys():
                self.state.save_dict = {"net_state": self.net, "optimizer_state": self.optimizer}
            else:
                self.state.save_dict = {"net_state": self.net, "embed_state": self.feat_dict, "optimizer_state": self.optimizer}
            self.callbacks.on_epoch_end()
            if self.state.early_stop:
                break

        self.callbacks.on_end()

    def _train_epoch(self, loader):

        self.net.train()
        if "h" not in self.g.ndata.keys():
            self.feat_dict.train()
        self.state.is_train = True
        self.state.num_batches = len(loader)
        self.callbacks.on_loader_start()

        for i, (input_nodes, output_nodes, mfgs) in enumerate(loader):
            self.state.num_samples_in_batch = output_nodes[self.g.category].size(0)
            self.state.iteration = i

            self.callbacks.on_batch_start()
            self._train_step(input_nodes, output_nodes, mfgs)
            self.callbacks.on_batch_end()

        if self.scheduler is not None:
            self.scheduler.step()

        self.callbacks.on_loader_end()

    def _train_step(self, input_nodes, output_nodes, mfgs):

        feat_dict = self._extract_feat(input_nodes)
        labels = self.g.labels[output_nodes[self.g.category].long()]
        logits = self.forward(None, mfgs, feat_dict, self.edge_weight_dict)[0]

        if self.g.num_classes == 2 and labels.dim() == 1:
            loss = self.criterion(logits, F.one_hot(labels.long(), num_classes=2).float())
        else:
            loss = self.criterion(logits, labels.long())

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.callbacks.on_after_backward()
        self.optimizer.step()

        self.state.losses = {"loss": loss.detach()}
        self.state.logits = {"logits": logits.detach()}
        self.state.labels = {"labels": labels}

    def _eval_epoch(self, loader):

        self.net.eval()
        if "h" not in self.g.ndata.keys():
            self.feat_dict.eval()
        self.state.is_train = False
        self.state.num_batches = len(loader)
        self.callbacks.on_loader_start()

        for i, (input_nodes, output_nodes, mfgs) in enumerate(loader):
            self.state.num_samples_in_batch = output_nodes[self.g.category].size(0)
            self.state.iteration = i

            self.callbacks.on_batch_start()
            self._eval_step(input_nodes, output_nodes, mfgs)
            self.callbacks.on_batch_end()

        self.callbacks.on_loader_end()

    @torch.no_grad()
    def _eval_step(self, input_nodes, output_nodes, mfgs):

        feat_dict = self._extract_feat(input_nodes)
        labels = self.g.labels[output_nodes[self.g.category].long()]
        logits = self.forward(None, mfgs, feat_dict, self.edge_weight_dict)[0]

        if self.g.num_classes == 2 and labels.dim() == 1:
            loss = self.criterion(logits, F.one_hot(labels.long(), num_classes=2).float())
        else:
            loss = self.criterion(logits, labels.long())

        self.state.losses = {"loss": loss}
        self.state.logits = {"logits": logits}
        self.state.labels = {"labels": labels}

    def test(self, net, g, full_batch=True, test_loader=None, callbacks=None, exp_dir=None, auto_restore=True):

        # prepare routines
        self.net = net
        self.g = g
        if "h" in self.g.ndata.keys():
            self.feat_dict = self.g.ndata["h"]
        else:
            from ..models.rgcn import RelGraphEmbed
            ntype2num = {ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes}
            if "feat" in g.ndata.keys():  # dblp
                feat_dict = RelGraphEmbed(ntype2num, 334)
            else:
                feat_dict = RelGraphEmbed(ntype2num, self.net.hidden_dim)
            self.feat_dict = feat_dict.to(self.device, non_blocking=True)
        if "edge_weight" in self.g.edata.keys():
            self.edge_weight_dict = self.g.edata["edge_weight"]
        else:
            self.edge_weight_dict = None
        if auto_restore:
            assert exp_dir is not None
            if "h" in self.g.ndata.keys():
                ckptio = CheckpointIO(ckpt_dir=exp_dir, device=self.device, net_state=self.net)
            else:
                ckptio = CheckpointIO(ckpt_dir=exp_dir, device=self.device, net_state=self.net, embed_state=self.feat_dict)
            ckptio.load()
        self.criterion = self._set_criterion(self.g.num_classes, self.device)
        self.callbacks = Callbacks(callbacks)
        self.state = RunnerState()
        self.callbacks.set_state(self.state)

        if full_batch:
            self.test_full_batch()
        else:
            assert test_loader is not None
            self.test_stochastic(test_loader)

        test_losses = {k: v.avg.item() for k, v in self.state.eval_loss_meters.items()}
        test_metrics = {k: v.avg.item() for k, v in self.state.eval_metric_meters.items()}
        return test_losses, test_metrics

    @torch.no_grad()
    def test_full_batch(self):

        self.callbacks.on_start()
        self.net.eval()
        if "h" not in self.g.ndata.keys():
            self.feat_dict.eval()
        self.state.is_train = False
        self.state.num_batches = 1
        self.callbacks.on_loader_start()
        self.state.num_samples_in_batch = self.g.test_idxs.size(0)
        self.state.iteration = 0
        self.callbacks.on_batch_start()

        labels = self.g.labels[self.g.test_idxs.long()]
        logits = self.forward(self.g, None, self.feat_dict, self.edge_weight_dict)[0][self.g.test_idxs.long()]

        if self.g.num_classes == 2 and labels.dim() == 1:
            loss = self.criterion(logits, F.one_hot(labels.long(), num_classes=2).float())
        else:
            loss = self.criterion(logits, labels.long())

        self.state.losses = {"loss": loss}
        self.state.logits = {"logits": logits}
        self.state.labels = {"labels": labels}

        self.callbacks.on_batch_end()
        self.callbacks.on_loader_end()
        self.callbacks.on_end()

    def test_stochastic(self, test_loader):

        self.callbacks.on_start()
        self.callbacks.on_loader_start()
        self._eval_epoch(test_loader)
        self.callbacks.on_loader_end()
        self.callbacks.on_end()

    def forward(self, g, mfgs, feat_dict, edge_weight_dict):
        if g is not None:
            if "h" in self.g.ndata.keys():
                return self.net(g=g, mfgs=None, feat_dict=feat_dict, edge_weight_dict=edge_weight_dict, out_key=self.g.category)
            else:
                feat_dict = {k: v for k, v in feat_dict().items()}
                if "feat" in g.ndata.keys():  # dblp
                    feat_dict["author"] = g.ndata["feat"]["author"]
                return self.net(g=g, mfgs=None, feat_dict=feat_dict, edge_weight_dict=edge_weight_dict, out_key=self.g.category)
        elif mfgs is not None:
            return self.net(g=None, mfgs=mfgs, feat_dict=feat_dict, edge_weight_dict=edge_weight_dict, out_key=self.g.category)
        else:
            assert False

    def _extract_feat(self, input_nodes):
        feat_dict = {}
        for ntype, nid in input_nodes.items():
            if "h" in self.g.ndata.keys():
                feat_dict[ntype] = self.feat_dict[ntype][nid.long()]
            else:
                feat_dict[ntype] = self.feat_dict()[ntype][nid.long()]
        return feat_dict

    @staticmethod
    def _configure_optimizers(params, opt_alg, lr, momentum=0.9, betas=(0.9, 0.999), weight_decay=5e-4,
                              milestones=(10000, 15000), gamma=0.1):
        if opt_alg == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif opt_alg == "adam":
            optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        else:
            assert False, f"Unknown optimizer: {opt_alg}"
        return optimizer, scheduler

    @staticmethod
    def _set_criterion(num_classes, device):
        if num_classes == 2:
            criterion = torch.nn.BCEWithLogitsLoss().to(device, non_blocking=True)
        else:
            criterion = torch.nn.CrossEntropyLoss().to(device, non_blocking=True)
        return criterion


class RunnerState(object):
    __isfrozen = False

    def __init__(self):

        # update every epoch
        self.epoch = 0
        self.save_dict = {}
        self.early_stop = False

        # update every loader
        self.num_batches = 0
        self.is_train = True
        self.train_loss_meters = defaultdict(AverageMeter)
        self.train_metric_meters = defaultdict(AverageMeter)
        self.eval_loss_meters = None
        self.eval_metric_meters = None

        # update every batch
        self.num_samples_in_batch = 0
        self.losses = {}
        self.logits = {}
        self.labels = {}
        self.iteration = 0
        self.loss_meters = defaultdict(AverageMeter)
        self.metric_meters = defaultdict(AverageMeter)

        self.__isfrozen = True

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(f"{self} is a frozen class")
        object.__setattr__(self, key, value)

    @property
    def epoch_log(self):
        return self.epoch + 1
