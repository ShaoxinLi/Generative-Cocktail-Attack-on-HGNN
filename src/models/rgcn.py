#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import torch
import torch.nn.functional as F
from functools import partial


class RGCN(torch.nn.Module):
    def __init__(self, meta_rels, input_dim, hidden_dim, output_dim, num_bases, num_hidden_layers=1,
                 dropout=0.0, use_bias=False, use_self_loop=False):
        super(RGCN, self).__init__()
        self.meta_rels = meta_rels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_bias = use_bias
        self.use_self_loop = use_self_loop
        self.etypes = list(set([meta_rel[1] for meta_rel in self.meta_rels]))
        self.etypes.sort()

        if num_bases < 0 or num_bases > len(self.etypes):
            self.num_bases = len(self.etypes)
        else:
            self.num_bases = num_bases

        self.layers = torch.nn.ModuleList()
        self.layers.append(
            RelGraphConvLayer(
                self.input_dim, self.hidden_dim, self.meta_rels, self.num_bases, use_weight=True,
                use_bias=self.use_bias, activation=F.relu, use_self_loop=self.use_self_loop,
                dropout=self.dropout
            )
        )
        for i in range(self.num_hidden_layers):
            self.layers.append(
                RelGraphConvLayer(
                    self.hidden_dim, self.hidden_dim, self.meta_rels, self.num_bases, use_weight=True,
                    use_bias=self.use_bias, activation=F.relu, use_self_loop=self.use_self_loop,
                    dropout=self.dropout
                )
            )
        self.layers.append(
            RelGraphConvLayer(
                self.hidden_dim, self.output_dim, self.meta_rels, self.num_bases, use_weight=True,
                use_bias=self.use_bias, activation=None, use_self_loop=self.use_self_loop,
                dropout=0.0
            )
        )

    def forward(self, g, mfgs, feat_dict, edge_weight_dict, out_key):
        if g is not None:  # full graph training
            for layer in self.layers:
                feat_dict, unaggre_feat_dict = layer(g, feat_dict, edge_weight_dict)
        elif mfgs is not None:  # stochastic training
            for layer, mfg in zip(self.layers, mfgs):
                if edge_weight_dict is not None:
                    new_edge_weight_dict = self._extract_edge_weights(mfg, edge_weight_dict)
                else:
                    new_edge_weight_dict = None
                feat_dict, unaggre_feat_dict = layer(mfg, feat_dict, new_edge_weight_dict)
        else:
            assert False
        if out_key is not None:
            return feat_dict[out_key], unaggre_feat_dict
        else:
            return feat_dict, unaggre_feat_dict

    @staticmethod
    def _extract_edge_weights(mfg, edge_weight_dict):
        new_edge_weight_dict = {}
        for meta_rel, eid in mfg.edata[dgl.EID].items():
            edge_weights = edge_weight_dict[meta_rel][eid.long()]
            new_edge_weight_dict[meta_rel] = edge_weights
        return new_edge_weight_dict


class RelGraphConvLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, meta_rels, num_bases, *, use_weight=True, use_bias=False, activation=None,
                 use_self_loop=False, dropout=0.0):

        super(RelGraphConvLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.meta_rels = meta_rels
        self.num_bases = num_bases
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.activation = activation
        self.use_self_loop = use_self_loop
        self.etypes = list(set([meta_rel[1] for meta_rel in self.meta_rels]))
        self.etypes.sort()

        self.conv = HeteroGraphConv(
            {meta_rel: dgl.nn.GraphConv(input_dim, output_dim, norm="both", weight=False, bias=False) for meta_rel in self.meta_rels}
        )

        self.use_basis = num_bases < len(self.etypes) and self.use_weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dgl.nn.WeightBasis((input_dim, output_dim), num_bases, len(self.etypes))
            else:
                self.weight = torch.nn.Parameter(torch.Tensor(len(self.etypes), input_dim, output_dim))
                torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain("relu"))
        if self.use_bias:
            self.bias_weight = torch.nn.Parameter(torch.Tensor(output_dim))
            torch.nn.init.zeros_(self.bias_weight)
        if self.use_self_loop:
            self.loop_weight = torch.nn.Parameter(torch.Tensor(input_dim, output_dim))
            torch.nn.init.xavier_uniform_(self.loop_weight, gain=torch.nn.init.calculate_gain("relu"))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, g, feat_dict, edge_weight_dict):
        with g.local_scope():
            if g.is_block:
                feat_dict_src = feat_dict
                feat_dict_dst = {k: v[: g.num_dst_nodes(k)] for k, v in feat_dict.items()}
            else:
                feat_dict_src = feat_dict_dst = feat_dict

            mod_kwargs = {meta_rel: {} for meta_rel in self.meta_rels}
            if self.use_weight:
                weight = self.basis() if self.use_basis else self.weight    # (num_etypes, input_dim, output_dim)
                for meta_rel in mod_kwargs.keys():
                    mod_kwargs[meta_rel]["weight"] = weight[self.etypes.index(meta_rel[1])]
            if edge_weight_dict is not None:
                norm = dgl.nn.EdgeWeightNorm(norm="both", eps=1e-6)
                for meta_rel, edge_weights in edge_weight_dict.items():
                    normed_edge_weights = norm(g[meta_rel], edge_weights)
                    mod_kwargs[meta_rel]["edge_weight"] = normed_edge_weights

            feat_dict, unaggre_feat_dict = self.conv(g, feat_dict, mod_kwargs=mod_kwargs)

        def _apply(ntype, h):
            if self.use_self_loop:
                h = h + torch.matmul(feat_dict_dst[ntype], self.loop_weight)
            if self.use_bias:
                h = h + self.bias_weight
            if self.activation is not None:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in feat_dict.items()}, unaggre_feat_dict


class HeteroGraphConv(torch.nn.Module):

    def __init__(self, mods, aggregate="sum"):
        super(HeteroGraphConv, self).__init__()
        self.mod_dict = mods
        mods = {str(k): v for k, v in mods.items()}
        self.mods = torch.nn.ModuleDict(mods)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, "set_allow_zero_in_degree", None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def _get_module(self, meta_rel):
        mod = self.mod_dict.get(meta_rel, None)
        if mod is not None:
            return mod
        raise KeyError("Cannot find module with relation %s" % meta_rel)

    def forward(self, g, feat_dict, mod_args=None, mod_kwargs=None):

        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {ntype: {} for ntype in g.dsttypes}

        if isinstance(feat_dict, tuple):
            feat_dict_src, feat_dict_dst = feat_dict
        elif g.is_block:
            feat_dict_src = feat_dict
            feat_dict_dst = {k: v[: g.num_dst_nodes(k)] for k, v in feat_dict.items()}
        else:
            feat_dict_src = feat_dict_dst = feat_dict

        for meta_rel in g.canonical_etypes:
            ntype_src, etype, ntype_dst = meta_rel
            meta_rel_graph = g[ntype_src, etype, ntype_dst]
            if ntype_src not in feat_dict_src or ntype_dst not in feat_dict_dst:
                continue
            conv = self._get_module(meta_rel)
            if "edge_weight" in mod_kwargs.get(meta_rel, {}).keys():
                if hasattr(conv, "_norm"):
                    conv._norm = "none"
            feat_dst = conv(
                meta_rel_graph,
                (feat_dict_src[ntype_src], feat_dict_dst[ntype_dst]),
                *mod_args.get(meta_rel, ()),
                **mod_kwargs.get(meta_rel, {})
            )
            outputs[ntype_dst][meta_rel] = feat_dst
        rsts = {}
        for ntype, feat_dict in outputs.items():
            if feat_dict:
                feat_list = list(feat_dict.values())
                rsts[ntype] = self.agg_fn(feat_list, ntype)
        return rsts, outputs


def _max_reduce_func(inputs, dim):
    return torch.max(inputs, dim=dim)[0]


def _min_reduce_func(inputs, dim):
    return torch.min(inputs, dim=dim)[0]


def _sum_reduce_func(inputs, dim):
    return torch.sum(inputs, dim=dim)


def _mean_reduce_func(inputs, dim):
    return torch.mean(inputs, dim=dim)


def _stack_agg_func(inputs, dsttype):  # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    return torch.stack(inputs, dim=1)


def _agg_func(inputs, dsttype, fn):  # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    stacked = torch.stack(inputs, dim=0)
    return fn(stacked, dim=0)


def get_aggregate_fn(agg):
    """Internal function to get the aggregation function for node data
    generated from different relations.
    Parameters
    ----------
    agg : str
        Method for aggregating node features generated by different relations.
        Allowed values are 'sum', 'max', 'min', 'mean', 'stack'.
    Returns
    -------
    callable
        Aggregator function that takes a list of tensors to aggregate
        and returns one aggregated tensor.
    """
    if agg == "sum":
        fn = _sum_reduce_func
    elif agg == "max":
        fn = _max_reduce_func
    elif agg == "min":
        fn = _min_reduce_func
    elif agg == "mean":
        fn = _mean_reduce_func
    elif agg == "stack":
        fn = None  # will not be called
    else:
        assert False
    if agg == "stack":
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)


class RelGraphEmbed(torch.nn.Module):
    def __init__(self, ntype2num, embed_dim):
        super(RelGraphEmbed, self).__init__()
        self.ntype2num = ntype2num
        self.embed_dim = embed_dim

        # create weight embeddings for each node for each relation
        self.embeds = torch.nn.ParameterDict()
        for ntype in ntype2num.keys():
            embeds = torch.nn.Parameter(torch.Tensor(ntype2num[ntype], self.embed_dim))
            torch.nn.init.xavier_uniform_(embeds, gain=torch.nn.init.calculate_gain("relu"))
            self.embeds[ntype] = embeds

    def forward(self, g=None):
        return self.embeds