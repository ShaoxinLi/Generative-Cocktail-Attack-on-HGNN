#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


class Accuracy(object):

    def __init__(self, topk=(1,)):
        self.name = "Acc@" + str(topk)
        self.topk = topk

    @torch.no_grad()
    def __call__(self, **kwargs):
        y_pred = kwargs["logits"]
        y_true = kwargs["labels"]
        maxk = max(self.topk)
        _, pred = y_pred.topk(maxk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(y_true.view(1, -1).expand_as(pred))

        res = []
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k / y_true.size(0) * 100.)
        return res
