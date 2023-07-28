#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import copy
import math
import random
import torch
import torch.nn.functional as F
import itertools
from collections import defaultdict
from .callbacks import Callbacks
from ..utils import seed_everything, AverageMeter, CheckpointIO
from ..losses.cw_loss import CWLoss
from sklearn.metrics import f1_score


class Attack(object):
    def __init__(self, opt_alg, lr, num_epochs, device, seed=None):
        super(Attack, self).__init__()
        self.opt_alg = opt_alg
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        self.seed = seed

    def attack(self, g, net, predictor, target_net, embeds=None, xi=10,
               search_scope=0, num_samplings=-1, batch_size=64, callbacks=None, lmd=1.0):

        # prepare routines
        self.g = g
        self.net = net
        self.predictor = predictor
        self.target_net = target_net
        if "h" in self.g.ndata.keys():
            self.feat_dict = self.g.ndata["h"]
        else:
            assert embeds is not None
            self.feat_dict = embeds()
        self.num_layers_of_target_net = self.target_net.num_layers
        self.xi = torch.tensor(xi, device=self.device)
        self.lmd = lmd

        params = itertools.chain(self.net.parameters(), self.predictor.parameters())
        self.optimizer, self.scheduler = self._configure_optimizers(params=params, opt_alg=self.opt_alg, lr=self.lr)
        self.criterion = self._set_criterion(self.g.num_classes, self.device)

        train_loader, val_loader = self.prepare_dataloader(
            self.g, search_scope=search_scope, num_samplings=num_samplings,
            is_train=True, num_epochs=self.num_epochs, batch_size=batch_size
        )

        seed_everything(self.seed)
        self.callbacks = Callbacks(callbacks)
        self.state = RunnerState()
        self.callbacks.set_state(self.state)

        # start training
        self.callbacks.on_start()
        for epoch in range(self.num_epochs):
            self.state.epoch = epoch
            self.callbacks.on_epoch_start()

            # train one epoch
            self._train_epoch(train_loader, batch_size)

            # val one epoch
            if val_loader is not None:
                self._eval_epoch(val_loader)

            if (epoch + 1) == 50:  # 10
                self.lmd *= 10
            if (epoch + 1) == 100:  # 100
                self.lmd *= 10
            if (epoch + 1) == 150:  # 1000
                self.lmd *= 10

            cur_iter = (epoch + 1) * math.ceil(len(g.train_idxs) / batch_size)
            cur_slope = 1.0 * 1.001 ** (cur_iter - 1)
            self.state.save_dict = {
                "net_state": self.net, "predictor_state": self.predictor,
                "optimizer_state": self.optimizer, "slope": cur_slope,
                "lmd": self.lmd,
            }
            self.callbacks.on_epoch_end()
            if self.state.early_stop:
                break

        self.callbacks.on_end()

    def _train_epoch(self, loader, batch_size):

        self.net.train()
        self.predictor.train()
        self.target_net.eval()
        self.state.is_train = True
        self.state.num_batches = len(loader)
        self.callbacks.on_loader_start()

        batch = []
        self.state.iteration = 0
        for i, (perturbed_g, flip_prob_dict, output_nodes, _) in enumerate(loader):
            batch.append((perturbed_g, flip_prob_dict, output_nodes))
            if (i + 1) % batch_size == 0 or (i + 1) == len(loader):
                self.state.num_samples_in_batch = len(batch)
                self.state.iteration += 1

                self.callbacks.on_batch_start()
                self._train_step(batch)
                self.callbacks.on_batch_end()

                batch = []

        if self.scheduler is not None:
            self.scheduler.step()

        self.callbacks.on_loader_end()

    def _train_step(self, batch):

        batch_logits, batch_labels, batch_flip_prob_dict = [], [], []
        for sample in batch:
            perturbed_g, flip_prob_dict, output_nodes = sample
            assert "edge_weight" in perturbed_g.edata.keys()
            edge_weight_dict = perturbed_g.edata["edge_weight"]
            label = self.g.labels[output_nodes[self.g.category].long()]
            logits = self.forward(perturbed_g, self.feat_dict, edge_weight_dict, self.g.category)[0][output_nodes[self.g.category].long()]

            batch_logits.append(logits)
            batch_labels.append(label)
            batch_flip_prob_dict.append(flip_prob_dict)

        batch_logits = torch.cat(batch_logits)
        batch_labels = torch.cat(batch_labels)

        # attack loss
        if self.g.num_classes == 2 and batch_labels.dim() == 1:
            attack_loss = self.criterion(batch_logits, F.one_hot(batch_labels.long(), num_classes=2).float())
        else:
            attack_loss = self.criterion(batch_logits, batch_labels.long())

        # xi loss
        xi_loss = torch.tensor(0.0, device=self.device)
        for flip_prob_dict in batch_flip_prob_dict:
            num_flipped_edges = 0
            for flip_probs in flip_prob_dict.values():
                num_flipped_edges += torch.norm(flip_probs, p=1)
            if num_flipped_edges > self.xi:
                xi_loss += num_flipped_edges - self.xi
        xi_loss /= self.state.num_samples_in_batch

        # backward
        total_loss = attack_loss + self.lmd * xi_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        self.state.losses = {
            "attack_loss": attack_loss.detach(),
            "xi_loss": xi_loss.detach(),
            "total_loss": total_loss.detach()
        }
        self.state.logits = {"logits": batch_logits.detach()}
        self.state.labels = {"labels": batch_labels}

    def _eval_epoch(self, loader):

        self.net.eval()
        self.predictor.eval()
        self.target_net.eval()
        self.state.is_train = False
        self.state.num_batches = len(loader)
        self.callbacks.on_loader_start()
        self.labels, self.logits = [], []
        self.rel_counter = {}

        for i, (perturbed_g, flip_prob_dict, output_nodes, num_remove_edge_dict) in enumerate(loader):
            self.state.num_samples_in_batch = output_nodes[self.g.category].size(0)
            self.state.iteration = i

            self.callbacks.on_batch_start()
            self._eval_step(perturbed_g, flip_prob_dict, output_nodes, num_remove_edge_dict)
            self.callbacks.on_batch_end()

        self.callbacks.on_loader_end()

        self.labels = torch.cat(self.labels, dim=0)
        self.logits = torch.cat(self.logits, dim=0)
        macro_f1_score = f1_score(self.labels.cpu(), self.logits.argmax(dim=1).cpu(), average="macro").item() * 100.
        micro_f1_score = f1_score(self.labels.cpu(), self.logits.argmax(dim=1).cpu(), average="micro").item() * 100.

        return macro_f1_score, micro_f1_score

    @torch.no_grad()
    def _eval_step(self, perturbed_g, flip_prob_dict, output_nodes, num_remove_edge_dict):

        assert "edge_weight" in perturbed_g.edata.keys()
        edge_weight_dict = perturbed_g.edata["edge_weight"]
        labels = self.g.labels[output_nodes[self.g.category].long()]
        logits = self.forward(perturbed_g, self.feat_dict, edge_weight_dict, self.g.category)[0][output_nodes[self.g.category].long()]

        self.labels.append(labels)
        self.logits.append(logits)
        if flip_prob_dict:
            for meta_rel, flip_probs in flip_prob_dict.items():
                if meta_rel not in self.rel_counter.keys():
                    d = {
                        "num_remove_edges": num_remove_edge_dict[meta_rel],
                        "num_add_edges": flip_probs.size(0) - num_remove_edge_dict[meta_rel]
                    }
                    self.rel_counter[meta_rel] = d
                else:
                    self.rel_counter[meta_rel]["num_remove_edges"] += num_remove_edge_dict[meta_rel]
                    num_add_edges = flip_probs.size(0) - num_remove_edge_dict[meta_rel]
                    self.rel_counter[meta_rel]["num_add_edges"] += num_add_edges

        # attack loss
        if self.g.num_classes == 2 and labels.dim() == 1:
            attack_loss = self.criterion(logits, F.one_hot(labels.long(), num_classes=2).float())
        else:
            attack_loss = self.criterion(logits, labels.long())

        # xi loss
        if flip_prob_dict:
            xi_loss = torch.tensor(0.0, device=self.device)
            num_flipped_edges = 0
            for flip_probs in flip_prob_dict.values():
                num_flipped_edges += torch.norm(flip_probs, p=1)
            if num_flipped_edges > self.xi:
                xi_loss += num_flipped_edges - self.xi
        else:
            xi_loss = torch.tensor(0., device=self.device)

        total_loss = attack_loss + self.lmd * xi_loss

        self.state.losses = {
            "attack_loss": attack_loss,
            "xi_loss": xi_loss,
            "total_loss": total_loss
        }
        self.state.logits = {"logits": logits}
        self.state.labels = {"labels": labels}
        self.state.flip_prob_dict = {meta_rel: filp_probs for meta_rel, filp_probs in flip_prob_dict.items()}

    def test(self, g, net, predictor, target_net, exp_dir, embeds=None, xi=10,
             search_scope=0, num_samplings=-1, callbacks=None, auto_restore=True):

        # prepare routines
        self.g = g
        self.net = net
        self.predictor = predictor
        self.target_net = target_net
        if "h" in self.g.ndata.keys():
            self.feat_dict = self.g.ndata["h"]
        else:
            assert embeds is not None
            self.feat_dict = embeds()
        self.num_layers_of_target_net = self.target_net.num_layers
        self.xi = xi
        self.slope = None
        self.lmd = None

        if auto_restore:
            ckptio = CheckpointIO(
                ckpt_dir=exp_dir, device=self.device, net_state=self.net,
                predictor_state=self.predictor, slope=self.slope, lmd=self.lmd
            )
            module_dict = ckptio.load()
            self.slope = module_dict["slope"]
            self.lmd = module_dict["lmd"]
        else:
            assert self.slope is not None
            assert self.lmd is not None

        test_loader = self.prepare_dataloader(
            self.g, search_scope=search_scope, num_samplings=num_samplings, is_train=False,
            test_slope=self.slope
        )

        self.criterion = self._set_criterion(self.g.num_classes, self.device)
        self.callbacks = Callbacks(callbacks)
        self.state = RunnerState()
        self.callbacks.set_state(self.state)

        self.callbacks.on_start()
        self.callbacks.on_loader_start()
        macro_f1_score, micro_f1_score = self._eval_epoch(test_loader)
        self.callbacks.on_loader_end()
        self.callbacks.on_end()

        test_losses = {k: v.avg.item() for k, v in self.state.eval_loss_meters.items()}
        test_metrics = {k: v.avg.item() for k, v in self.state.eval_metric_meters.items()}
        return test_losses, test_metrics, macro_f1_score, micro_f1_score, self.rel_counter

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
        criterion = CWLoss(kappa=5., num_classes=num_classes)
        criterion = criterion.to(device, non_blocking=True)
        return criterion

    def prepare_dataloader(self, g, search_scope=1, num_samplings=-1, is_train=True,
                           num_epochs=100, batch_size=32, test_slope=1000.):

        if is_train:
            num_training_batches = math.ceil(len(g.train_idxs) / batch_size)
            sampler_train = AttackNeighborSampler(
                net=self.net, predictor=self.predictor, num_layers_of_target_net=self.num_layers_of_target_net,
                feat_dict=self.feat_dict, xi=None, device=self.device, search_scope=search_scope,
                num_samplings=num_samplings, mode="train", num_epochs=num_epochs,
                num_samples=len(g.train_idxs), num_training_batches=num_training_batches, batch_size=batch_size
            )
            train_loader = dgl.dataloading.DataLoader(
                graph=g, indices={g.category: g.train_idxs}, graph_sampler=sampler_train,
                device=self.device, batch_size=1, shuffle=True, num_workers=0
            )
            sampler_val = AttackNeighborSampler(
                net=self.net, predictor=self.predictor, num_layers_of_target_net=self.num_layers_of_target_net,
                feat_dict=self.feat_dict, xi=self.xi, device=self.device, search_scope=search_scope,
                num_samplings=num_samplings, mode="val", num_epochs=num_epochs, num_samples=len(g.val_idxs),
                num_training_batches=num_training_batches,
            )
            val_loader = dgl.dataloading.DataLoader(
                graph=g, indices={g.category: g.val_idxs}, graph_sampler=sampler_val,
                device=self.device, batch_size=1, shuffle=False, num_workers=0
            )
            return train_loader, val_loader
        else:
            sampler_test = AttackNeighborSampler(
                net=self.net, predictor=self.predictor, num_layers_of_target_net=self.num_layers_of_target_net,
                feat_dict=self.feat_dict, xi=self.xi, device=self.device, search_scope=search_scope,
                num_samplings=num_samplings, mode="test", test_slope=test_slope
            )
            test_loader = dgl.dataloading.DataLoader(
                graph=g, indices={g.category: g.test_idxs}, graph_sampler=sampler_test,
                device=self.device, batch_size=1, shuffle=False, num_workers=0
            )
            return test_loader

    def forward(self, perturbed_g, feat_dict, edge_weight_dict, out_key):
        if "h" not in self.g.ndata.keys():
            feat_dict = {k: v.data for k, v in feat_dict.items()}
        return self.target_net(g=perturbed_g, mfgs=None, feat_dict=feat_dict,
                               edge_weight_dict=edge_weight_dict, out_key=out_key)


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
        self.flip_prob_dict = None

        self.__isfrozen = True

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(f"{self} is a frozen class")
        object.__setattr__(self, key, value)

    @property
    def epoch_log(self):
        return self.epoch + 1


class AttackNeighborSampler(dgl.dataloading.Sampler):
    def __init__(self, net, predictor, num_layers_of_target_net, feat_dict, xi, device,
                 search_scope=0, num_samplings=-1, mode="train", num_epochs=100, num_samples=100,
                 num_training_batches=100, batch_size=32, test_slope=1.0):
        super(AttackNeighborSampler, self).__init__()
        assert search_scope <= num_layers_of_target_net
        self.net = net
        self.predictor = predictor
        self.num_layers_of_target_net = num_layers_of_target_net
        self.feat_dict = feat_dict
        self.xi = xi
        self.device = device
        self.search_scope = search_scope
        self.num_samplings = num_samplings
        self.mode = mode
        self.test_slope = test_slope

        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.num_training_batches = num_training_batches
        self.batch_size = batch_size
        self.num_iters = self.num_training_batches * self.num_epochs

        self.cur_iter = 1
        self.counter = 0

        if self.mode == "train":
            self.net.train()
            self.predictor.train()
        else:
            self.net.eval()
            self.predictor.eval()

    def sample(self, g, seed_nodes):
        self.g = g
        self.seed_nodes = seed_nodes

        if self.mode == "train":
            self.cur_slope = 1.0 * 1.001 ** (self.cur_iter - 1)
            if self.counter != 0 and self.counter % self.batch_size == 0 or self.counter == self.num_samples:     # one iter
                self.cur_iter += 1
                self.cur_slope = 1.0 * 1.001 ** (self.cur_iter - 1)
                if self.counter == self.num_samples:      # reset the counter at the end of an epoch
                    self.counter = 1
                else:
                    self.counter += 1
            else:
                self.counter += 1
        elif self.mode == "val":
            if self.counter == 0:
                if self.cur_iter == 1:
                    self.cur_iter = self.cur_iter + self.num_training_batches - 1
                else:
                    self.cur_iter += self.num_training_batches
                self.cur_slope = 1.0 * 1.001 ** (self.cur_iter - 1)
            self.counter += 1
            if self.counter == self.num_samples:
                self.counter = 0
        elif self.mode == "test":
            self.cur_slope = self.test_slope
        else:
            assert False
        exist_edge_dict = self._retrieve_exist_edges(self.g, self.seed_nodes, self.num_layers_of_target_net)
        non_exist_edge_dict = self._retrieve_non_exist_edges(
            self.g, self.seed_nodes, exist_edge_dict, self.search_scope, self.num_samplings, self.device
        )

        # get the fully connected graph
        g_, edge_weights_dict = self._add_edges(self.g, exist_edge_dict, non_exist_edge_dict, self.device)

        # perturb the graph
        pert_edge_weights_dict, flip_prob_dict, num_remove_edge_dict = self.perturb(exist_edge_dict, non_exist_edge_dict, edge_weights_dict)

        for edge_weight in pert_edge_weights_dict.values():
            assert torch.max(edge_weight.data) <= 1.0
            assert torch.min(edge_weight.data) >= 0.0
        g_.edata["edge_weight"] = pert_edge_weights_dict

        return g_, flip_prob_dict, seed_nodes, num_remove_edge_dict

    def perturb(self, exist_edge_dict, non_exist_edge_dict, edge_weights_dict):

        # RGCN forward
        feat_dict, unaggre_feat_dict = self.net(g=self.g, mfgs=None, feat_dict=self.feat_dict, edge_weight_dict=None, out_key=None)

        flip_prob_dict, flip_indicator_dict, pert_eid_dict = {}, {}, {}
        for meta_rel, (nids_src, nids_dst, eids) in exist_edge_dict.items():
            feats_src = feat_dict[meta_rel[0]][nids_src.long()]
            feats_dst = unaggre_feat_dict[meta_rel[-1]][meta_rel][nids_dst.long()]
            feats = feats_src + feats_dst
            flip_probs, flip_indicators = self.predictor((feats, self.cur_slope))
            flip_probs = flip_probs.squeeze(dim=-1)
            flip_indicators = flip_indicators.squeeze(dim=-1)

            flip_prob_dict[meta_rel] = flip_probs
            flip_indicator_dict[meta_rel] = flip_indicators
            pert_eid_dict[meta_rel] = eids.long()

        for meta_rel, (nids_src, nids_dst) in non_exist_edge_dict.items():
            feats_src = feat_dict[meta_rel[0]][nids_src.long()]
            feats_dst = unaggre_feat_dict[meta_rel[-1]][meta_rel][nids_dst.long()]
            feats = (feats_src + feats_dst)
            flip_probs, flip_indicators = self.predictor((feats, self.cur_slope))
            flip_probs = flip_probs.squeeze(dim=-1)
            flip_indicators = flip_indicators.squeeze(dim=-1)

            num_total_edges = edge_weights_dict[meta_rel].size(0)
            num_non_exist_edges = nids_src.size(0)
            if meta_rel not in flip_prob_dict.keys():
                flip_prob_dict[meta_rel] = flip_probs
                flip_indicator_dict[meta_rel] = flip_indicators
                pert_eid_dict[meta_rel] = torch.arange(num_total_edges - num_non_exist_edges, num_total_edges, device=self.device)
            else:
                p = flip_prob_dict[meta_rel]
                p = torch.cat([p, flip_probs])
                flip_prob_dict[meta_rel] = p

                q = flip_indicator_dict[meta_rel]
                q = torch.cat([q, flip_indicators])
                flip_indicator_dict[meta_rel] = q

                n = pert_eid_dict[meta_rel]
                n = torch.cat([n, torch.arange(num_total_edges - num_non_exist_edges, num_total_edges, device=self.device)])
                pert_eid_dict[meta_rel] = n

        if self.mode in ["val", "test"]:       # for evaluation
            real_flip_prob_dict, real_pert_eid_dict = {}, {}
            flip_probs, lookup_table, accumulates = [], [], []
            for meta_rel, probs in flip_prob_dict.items():
                flip_probs.append(probs)
                lookup_table.append((meta_rel, probs.size(0)))
                if not accumulates:
                    accumulates.append(probs.size(0))
                else:
                    accumulates.append(probs.size(0) + accumulates[-1])
            flip_probs = torch.cat(flip_probs)

            real_eval_top_k = self.xi if flip_probs.size(0) >= self.xi else flip_probs.size(0)
            top_k_flip_probs, top_k_flip_prob_idxs = torch.topk(flip_probs, real_eval_top_k)       # wo choose to flip the top-k edges
            top_k_flip_prob_idxs = [idx for i, idx in enumerate(top_k_flip_prob_idxs) if top_k_flip_probs[i] > 1e-2]
            top_k_flip_prob_idxs = torch.as_tensor(top_k_flip_prob_idxs, device=self.device)

            for idx in top_k_flip_prob_idxs:
                for i, v in enumerate(accumulates):
                    if idx < v:
                        meta_rel = lookup_table[i][0]
                        idx_in_meta_rel = idx if i == 0 else idx - accumulates[i - 1]
                        if meta_rel not in real_flip_prob_dict.keys():
                            real_flip_prob_dict[meta_rel] = [1.0]
                            real_pert_eid_dict[meta_rel] = [pert_eid_dict[meta_rel][idx_in_meta_rel]]
                        else:
                            real_flip_prob_dict[meta_rel].append(1.0)
                            real_pert_eid_dict[meta_rel].append(pert_eid_dict[meta_rel][idx_in_meta_rel])
                        break
            for meta_rel, probs in real_flip_prob_dict.items():
                real_flip_prob_dict[meta_rel] = torch.as_tensor(probs, device=self.device)
            for meta_rel, eids in real_pert_eid_dict.items():
                real_pert_eid_dict[meta_rel] = torch.as_tensor(eids, device=self.device)
            flip_indicator_dict = real_flip_prob_dict
            pert_eid_dict = real_pert_eid_dict

        num_remove_edge_dict = {}
        for meta_rel, flip_probs in flip_indicator_dict.items():
            A = edge_weights_dict[meta_rel][pert_eid_dict[meta_rel]]
            if self.mode in ["val", "test"]:
                if meta_rel not in num_remove_edge_dict.keys():
                    num_remove_edge_dict[meta_rel] = A.sum()
                else:
                    num_remove_edge_dict[meta_rel] = num_remove_edge_dict[meta_rel] + A.sum()
            edge_weights_dict[meta_rel][pert_eid_dict[meta_rel]] = A + (1 - A - A) * flip_probs
        return edge_weights_dict, flip_indicator_dict, num_remove_edge_dict

    @staticmethod
    def _retrieve_exist_edges(g, seed_nodes, num_layers):
        exist_edge_dict = {}
        for _ in range(num_layers):
            sg = dgl.sampling.sample_neighbors(g, seed_nodes, fanout=-1, edge_dir="in")
            src_nodes = {}
            for meta_rel in sg.canonical_etypes:
                if sg.num_edges(meta_rel) != 0:
                    eids_in_rel_g = g.edge_ids(*sg.edges(etype=meta_rel), etype=meta_rel)
                    if meta_rel not in exist_edge_dict.keys():
                        exist_edge_dict[meta_rel] = *sg.edges(etype=meta_rel), eids_in_rel_g
                    else:
                        nids_src, nids_dst, eids = exist_edge_dict[meta_rel]
                        nids_src = torch.cat([nids_src, sg.edges(etype=meta_rel)[0]])
                        nids_dst = torch.cat([nids_dst, sg.edges(etype=meta_rel)[1]])
                        eids = torch.cat([eids, eids_in_rel_g])
                        exist_edge_dict[meta_rel] = nids_src, nids_dst, eids
                    if meta_rel[0] not in src_nodes.keys():
                        src_nodes[meta_rel[0]] = sg.edges(etype=meta_rel)[0]
                    else:
                        nids_src = src_nodes[meta_rel[0]]
                        nids_src = torch.cat([nids_src, sg.edges(etype=meta_rel)[0]])
                        src_nodes[meta_rel[0]] = nids_src
            seed_nodes = src_nodes
        return exist_edge_dict

    @staticmethod
    def _retrieve_non_exist_edges(g, seed_nodes, exist_edge_dict, search_scope, num_samplings, device):
        non_exist_edge_dict = {}
        if search_scope < 1:       # we only delete the existing edges
            return non_exist_edge_dict
        for _ in range(search_scope):
            src_nodes = {}
            for meta_rel in g.canonical_etypes:
                if meta_rel[-1] in seed_nodes.keys():
                    num_nodes_src = g.num_src_nodes(ntype=meta_rel[0])

                    # get all possible edges
                    nids_dst = seed_nodes[meta_rel[-1]].cpu().numpy()
                    possible_edges = {(nid_src, nid_dst) for nid_dst in nids_dst for nid_src in range(num_nodes_src)}

                    # get existing edges
                    if meta_rel in exist_edge_dict.keys():
                        nids_src_exist, nids_dst_exist, _ = exist_edge_dict[meta_rel]
                        nids_src_exist = nids_src_exist.cpu().numpy()
                        nids_dst_exist = nids_dst_exist.cpu().numpy()
                        exist_edges = {(nid_src, nid_dst) for nid_src, nid_dst in zip(nids_src_exist, nids_dst_exist)}
                    else:
                        exist_edges = set()

                    # get non-existing edges
                    non_exist_edges = possible_edges - exist_edges

                    # do sampling
                    if non_exist_edges:
                        if 0 < num_samplings < len(non_exist_edges):
                            non_exist_edges = random.sample(non_exist_edges, num_samplings)
                        nids_src_nonexist = [nid_src for (nid_src, nid_dst) in non_exist_edges]
                        nids_dst_nonexist = [nid_dst for (nid_src, nid_dst) in non_exist_edges]
                        nids_src_nonexist = torch.as_tensor(nids_src_nonexist, device=device, dtype=torch.int32)
                        nids_dst_nonexist = torch.as_tensor(nids_dst_nonexist, device=device, dtype=torch.int32)

                    if non_exist_edges and meta_rel not in non_exist_edge_dict.keys():
                        non_exist_edge_dict[meta_rel] = nids_src_nonexist, nids_dst_nonexist
                    elif non_exist_edges:
                        nids_src, nids_dst = non_exist_edge_dict[meta_rel]
                        nids_src = torch.cat([nids_src, nids_src_nonexist])
                        nids_dst = torch.cat([nids_dst, nids_dst_nonexist])
                        non_exist_edge_dict[meta_rel] = nids_src, nids_dst

                    if meta_rel[0] not in src_nodes.keys():
                        if meta_rel in exist_edge_dict.keys() and non_exist_edges:
                            nids_src_exist = torch.as_tensor(nids_src_exist, device=device, dtype=torch.int32)
                            src_nodes[meta_rel[0]] = torch.unique(torch.cat([nids_src_exist, nids_src_nonexist]))
                        elif meta_rel in exist_edge_dict.keys():
                            src_nodes[meta_rel[0]] = torch.as_tensor(nids_src_exist, device=device, dtype=torch.int32)
                        elif non_exist_edges:
                            src_nodes[meta_rel[0]] = nids_src_nonexist
                        else:
                            assert False
                    else:
                        nids_src = src_nodes[meta_rel[0]]
                        if meta_rel in exist_edge_dict.keys() and non_exist_edges:
                            nids_src_exist = torch.as_tensor(nids_src_exist, device=device, dtype=torch.int32)
                            nids_src = torch.unique(torch.cat([nids_src, nids_src_exist, nids_src_nonexist]))
                            src_nodes[meta_rel[0]] = nids_src
                        elif meta_rel in exist_edge_dict.keys():
                            nids_src_exist = torch.as_tensor(nids_src_exist, device=device, dtype=torch.int32)
                            nids_src = torch.unique(torch.cat([nids_src, nids_src_exist]))
                            src_nodes[meta_rel[0]] = nids_src
                        elif non_exist_edges:
                            nids_src = torch.unique(torch.cat([nids_src, nids_src_nonexist]))
                            src_nodes[meta_rel[0]] = nids_src
                        else:
                            assert False
            seed_nodes = src_nodes
        return non_exist_edge_dict

    @staticmethod
    def _add_edges(g, exist_edge_dict, non_exist_edge_dict, device):
        g_ = copy.deepcopy(g)

        # init edge weights
        edge_weights_dict = {}
        for meta_rel in g_.canonical_etypes:
            if meta_rel in exist_edge_dict.keys() and meta_rel in non_exist_edge_dict.keys():
                num_exist_edges = g_.num_edges(meta_rel)
                num_non_exist_edges = non_exist_edge_dict[meta_rel][0].size(0)
                edge_weights_dict[meta_rel] = torch.hstack((
                    torch.ones(num_exist_edges, device=device),
                    torch.zeros(num_non_exist_edges, device=device)
                ))
            elif meta_rel in exist_edge_dict.keys():
                num_exist_edges = g_.num_edges(meta_rel)
                edge_weights_dict[meta_rel] = torch.ones(num_exist_edges, device=device)
            elif meta_rel in non_exist_edge_dict.keys():
                num_exist_edges = g_.num_edges(meta_rel)        # caution here!
                num_non_exist_edges = non_exist_edge_dict[meta_rel][0].size(0)
                edge_weights_dict[meta_rel] = torch.hstack((
                    torch.ones(num_exist_edges, device=device),
                    torch.zeros(num_non_exist_edges, device=device)
                ))
            else:
                num_exist_edges = g_.num_edges(meta_rel)
                edge_weights_dict[meta_rel] = torch.ones(num_exist_edges, device=device)

        # add edges
        if non_exist_edge_dict:
            for meta_rel, (nids_src, nids_dst) in non_exist_edge_dict.items():
                g_.add_edges(nids_src, nids_dst, etype=meta_rel)
        return g_, edge_weights_dict

