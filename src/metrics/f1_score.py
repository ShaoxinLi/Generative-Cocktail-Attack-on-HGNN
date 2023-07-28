#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from sklearn.metrics import f1_score


class MacroF1Score(object):

    def __init__(self):
        self.name = "MacroF1"

    @torch.no_grad()
    def __call__(self, **kwargs):
        y_pred = kwargs["logits"]
        y_true = kwargs["labels"]
        score = f1_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average="macro")
        return score * 100.


class MicroF1Score(object):

    def __init__(self):
        self.name = "MicroF1"

    @torch.no_grad()
    def __call__(self, **kwargs):
        y_pred = kwargs["logits"]
        y_true = kwargs["labels"]
        score = f1_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average="micro")
        return score * 100.