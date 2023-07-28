#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class RoundFunctionST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


RoundST = RoundFunctionST.apply


class DeterministicBinaryActivation(torch.nn.Module):

    def __init__(self):
        super(DeterministicBinaryActivation, self).__init__()
        self.act = torch.nn.Sigmoid()
        self.binarizer = RoundST

    def forward(self, input):
        x, slope = input
        x = self.act(slope * x)
        x_bin = self.binarizer(x)
        return x, x_bin


class Predictor(torch.nn.Module):
    def __init__(self, input_dim):
        super(Predictor, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.fc2 = torch.nn.Linear(input_dim, int(input_dim / 2))
        self.fc3 = torch.nn.Linear(int(input_dim / 2), 1)
        self.act = DeterministicBinaryActivation()

    def forward(self, input):
        x, slope = input
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out, out_bin = self.act((out, slope))
        return out, out_bin
