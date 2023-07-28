#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class CWLoss(torch.nn.Module):
    def __init__(self, kappa, num_classes):
        super(CWLoss, self).__init__()
        self.kappa = kappa
        self.num_classes = num_classes

    def forward(self, logits, labels):
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes)
        class_logits = (one_hot_labels * logits).sum(1)
        non_class_logits = ((1 - one_hot_labels) * logits - one_hot_labels * 10000000.).max(1)[0].detach()
        loss = torch.clamp(class_logits - non_class_logits, min=-self.kappa).mean()
        return loss
