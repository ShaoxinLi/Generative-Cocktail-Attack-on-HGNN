#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .rgcn import RGCN
from .hgt import HGT
from .simple_hgn import SimpleHGN
from .hetsann import HetSANN
from .han import HAN
from .gtn import GTN
from .mhnf import MHNF
from .predictor import Predictor


def get_network(net_arch, input_dim, hidden_dim, output_dim, meta_rels=None, dataset_name=None):
    """Get a network"""

    if net_arch == "rgcn":
        net = RGCN(
            meta_rels=meta_rels, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            num_bases=-1, num_hidden_layers=0, dropout=0.0, use_bias=False,
            use_self_loop=False
        )
        net.num_layers = 2
    elif net_arch == "hgt":
        net = HGT(
            meta_rels=meta_rels, input_dim=input_dim, hidden_dim=hidden_dim,
            output_dim=output_dim, num_layers=2, num_heads=4, use_norm=True
        )
        net.num_layers = 2
    elif net_arch == "simple_hgn":
        net = SimpleHGN(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            edge_dim=64, meta_rels=meta_rels, num_heads_list=[8, 8, 1], num_layers=3,
            feat_drop=0.2, negative_slope=0.05, residual=True, beta=0.05
        )
        net.num_layers = 3
    elif net_arch == "hetsann":
        net = HetSANN(
            input_dim=input_dim, output_dim=output_dim, meta_rels=meta_rels,
            num_heads=16, num_layers=2, dropout=0.2, negative_slope=0.2, residual=True
        )
        net.num_layers = 2
    elif net_arch == "han":
        assert dataset_name is not None
        if dataset_name == "acm":
            meta_paths = [
                [("paper", "paper-author", "author"), ("author", "author-paper", "paper")],
                [("paper", "paper-subject", "subject"), ("subject", "subject-paper", "paper")]
            ]
        elif dataset_name == "imdb":
            meta_paths = [
                [("movie", "movie-director", "director"), ("director", "director-movie", "movie")],
                [("movie", "movie-actor", "actor"), ("actor", "actor-movie", "movie")]
            ]
        elif dataset_name == "dblp":
            meta_paths = [
                [("author", "ap", "paper"), ("paper", "pa", "author")],
                [("author", "ap", "paper"), ("paper", "pt", "term"), ("term", "tp", "paper"), ("paper", "pa", "author")],
                [("author", "ap", "paper"), ("paper", "pc", "conf"), ("conf", "cp", "paper"), ("paper", "pa", "author")]
            ]
        elif dataset_name == "dblp_paper":
            meta_paths = [
                [("paper", "pa", "author"), ("author", "ap", "paper")],
                [("paper", "pc", "conf"), ("conf", "cp", "paper")],
                [("paper", "pt", "term"), ("term", "tp", "paper")]
            ]
        else:
            assert False
        net = HAN(
            meta_paths=meta_paths, input_dim=input_dim, hidden_dim=hidden_dim,
            output_dim=output_dim, num_layers=2, num_heads=8, dropout=0.6
        )
        net.num_layers = 2
    elif net_arch == "gtn":
        net = GTN(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_meta_rels=len(meta_rels),
            num_meta_paths=2, num_layers=2, identity=True
        )
        net.num_layers = 2
    elif net_arch == "mhnf":
        net = MHNF(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_meta_rels=len(meta_rels),
            num_meta_paths=2, num_layers=2, identity=False
        )
        net.num_layers = 2
    else:
        assert False, f"Net arch {net_arch} is not supported yet."
    return net


def set_parameter_requires_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad


def get_num_parameters(net):
    return sum(p.numel() for p in net.parameters())


def get_num_trainable_parameters(net):
    net_parameters = filter(lambda p: p.requires_grad is True, net.parameters())
    return sum([np.prod(p.size()) for p in net_parameters])


def get_num_non_trainable_parameters(net):
    net_parameters = filter(lambda p: p.requires_grad is False, net.parameters())
    return sum([np.prod(p.size()) for p in net_parameters])