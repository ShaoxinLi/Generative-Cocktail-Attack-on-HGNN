#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import dgl.function as fn
import dgl.nn.pytorch
import torch
import torch.nn.functional as F


class HetSANN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, meta_rels, num_heads, num_layers, dropout, negative_slope, residual):
        super(HetSANN, self).__init__()
        self.num_layers = num_layers
        self.het_layers = torch.nn.ModuleList()
        self.activation = F.elu
        num_etypes = len([meta_rel[1] for meta_rel in meta_rels])
        ntypes_src = [meta_rel[0] for meta_rel in meta_rels]
        ntypes_dst = [meta_rel[-1] for meta_rel in meta_rels]
        ntypes = list(set(ntypes_src + ntypes_dst))
        ntypes.sort()
        self.ntypes = ntypes

        self.input_dim = input_dim
        self.hidden_dim = self.input_dim // num_heads
        self.output_dim = output_dim

        self.het_layers.append(
            HetSANNConv(self.input_dim, self.hidden_dim, num_heads, num_etypes, dropout,
                        negative_slope, False, self.activation)
        )
        for l in range(1, self.num_layers - 1):
            self.het_layers.append(
                HetSANNConv(int(self.hidden_dim * num_heads), self.hidden_dim, num_heads, num_etypes, dropout,
                            negative_slope, residual, self.activation)
            )
        self.het_layers.append(
            HetSANNConv(int(self.hidden_dim * num_heads), self.output_dim, 1, num_etypes, dropout,
                        negative_slope, residual, None)
        )

    def forward(self, g, mfgs, feat_dict, edge_weight_dict, out_key):
        if g is not None:
            with g.local_scope():
                g.ndata["h"] = feat_dict
                if edge_weight_dict is not None:
                    g.edata["ew"] = edge_weight_dict
                    g_homo = dgl.to_homogeneous(g, ndata=["h"], edata=["ew"])
                    feat = g_homo.ndata["h"]
                    edge_weight = g_homo.edata["ew"]
                else:
                    g_homo = dgl.to_homogeneous(g, ndata=["h"])
                    feat = g_homo.ndata["h"]
                    edge_weight = None
                for l in range(self.num_layers):
                    feat = self.het_layers[l](g_homo, feat, edge_weight, g_homo.ndata[dgl.NTYPE], g_homo.edata[dgl.ETYPE], True)
                new_feat_dict = {}
                for index, ntype in enumerate(g.ntypes):
                    new_feat_dict[ntype] = feat[torch.where(g_homo.ndata[dgl.NTYPE] == index)]
        elif mfgs is not None:
            feat = feat_dict
            edge_weight = edge_weight_dict
            for layer, mfg in zip(self.het_layers, mfgs):
                feat = layer(mfg, feat, edge_weight, mfg.ndata[dgl.NTYPE][dgl.NID], mfg.edata[dgl.ETYPE], False)
            new_feat_dict = {}
            for index, ntype in enumerate(self.ntypes):
                new_feat_dict[ntype] = feat[torch.where(mfg.ndata[dgl.NTYPE][:mfg.num_dst_nodes()] == index)]
        else:
            assert False
        if out_key is not None:
            return new_feat_dict[out_key],
        else:
            return new_feat_dict,


class HetSANNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_etypes, dropout, negative_slope,
                 residual, activation):
        super(HetSANNConv, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = dgl.nn.pytorch.TypedLinear(self.input_dim, self.output_dim * self.num_heads, num_etypes)
        self.a_l = dgl.nn.pytorch.TypedLinear(self.output_dim * self.num_heads, self.output_dim * self.num_heads, num_etypes)
        self.a_r = dgl.nn.pytorch.TypedLinear(self.output_dim * self.num_heads, self.output_dim * self.num_heads, num_etypes)

        self.dropout = torch.nn.Dropout(dropout)
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope)
        if residual:
            self.residual = torch.nn.Linear(self.input_dim, self.output_dim * self.num_heads)
        else:
            self.register_buffer("residual", None)

        self.activation = activation

    def forward(self, g_homo, feat, edge_weight, ntype_idxs, etype_idxs, presorted=False):
        g_homo.srcdata["h"] = feat      # num_nodes, input_dim
        g_homo.apply_edges(fn.copy_u("h", "m"))
        h = g_homo.edata["m"]           # num_edges, input_dim
        h = self.W(h, etype_idxs, presorted)     # num_edges, output_dim * num_heads
        h = self.dropout(h)
        h = h.view(-1, self.num_heads, self.output_dim)     # num_edges, num_heads, output_dim

        h_l = self.a_l(h.view(-1, self.num_heads * self.output_dim), etype_idxs, presorted)
        h_l = h_l.view(-1, self.num_heads, self.output_dim).sum(dim=-1)     # num_edges, num_heads
        h_r = self.a_r(h.view(-1, self.num_heads * self.output_dim), etype_idxs, presorted)
        h_r = h_r.view(-1, self.num_heads, self.output_dim).sum(dim=-1)  # num_edges, num_heads

        attention = self.leakyrelu(h_l + h_r)       # num_edges, num_heads
        # attention = dgl.nn.functional.edge_softmax(g_homo, attention)       # num_edges, num_heads
        attention = torch.sigmoid(attention)        # num_edges, num_heads

        if edge_weight is not None:
            edge_weight = edge_weight.unsqueeze(-1)     # num_edges, 1
            attention = attention * edge_weight       # num_edges, num_heads

        with g_homo.local_scope():
            h = h.permute(0, 2, 1).contiguous()     # num_edges, output_dim, num_heads
            g_homo.edata["alpha"] = h * attention.reshape(-1, 1, self.num_heads)    # num_edges, output_dim, num_heads
            g_homo.update_all(fn.copy_e("alpha", "w"), fn.sum("w", "emb"))
            feat_output = g_homo.dstdata["emb"].view(-1, self.output_dim * self.num_heads)     # num_nodes, output_dim * num_heads

        if g_homo.is_block:
            feat = feat[:g_homo.num_dst_nodes()]
        if self.residual:
            res = self.residual(feat)
            feat_output += res
        if self.activation is not None:
            feat_output = self.activation(feat_output)

        return feat_output
