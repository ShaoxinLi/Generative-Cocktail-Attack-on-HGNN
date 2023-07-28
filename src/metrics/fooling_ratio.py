#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


class FoolingRatio(object):

    def __init__(self):
        self.name = "FoolingRatio"

    @torch.no_grad()
    def __call__(self, **kwargs):
        y_pred = kwargs["logits"]
        y_true = kwargs["labels"]
        correct = y_pred.argmax(dim=1).eq(y_true).sum()
        return (1. - correct / y_true.size(0)) * 100.

