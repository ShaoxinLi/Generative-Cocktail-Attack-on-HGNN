#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import dgl.function as fn
import dgl.nn.pytorch
import torch
import torch.nn.functional as F


class SimpleHGN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, meta_rels, num_heads_list, num_layers,
                 feat_drop, negative_slope, residual, beta):
        super(SimpleHGN, self).__init__()
        self.num_layers = num_layers
        self.hgn_layers = torch.nn.ModuleList()
        self.activation = F.elu
        num_etypes = len([meta_rel[1] for meta_rel in meta_rels])
        ntypes_src = [meta_rel[0] for meta_rel in meta_rels]
        ntypes_dst = [meta_rel[-1] for meta_rel in meta_rels]
        ntypes = list(set(ntypes_src + ntypes_dst))
        ntypes.sort()
        self.ntypes = ntypes

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.hgn_layers.append(
            SimpleHGNConv(self.input_dim, self.hidden_dim, edge_dim, num_heads_list[0], num_etypes, feat_drop,
                          negative_slope, False, self.activation, beta)
        )
        for l in range(1, self.num_layers - 1):
            self.hgn_layers.append(
                SimpleHGNConv(self.hidden_dim * num_heads_list[l - 1], self.hidden_dim, edge_dim, num_heads_list[l],
                              num_etypes, feat_drop, negative_slope, residual, self.activation, beta)
            )
        self.hgn_layers.append(
            SimpleHGNConv(self.hidden_dim * num_heads_list[-2], self.output_dim, edge_dim, num_heads_list[-1], num_etypes,
                          feat_drop, negative_slope, residual, None, beta)
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
                    feat = self.hgn_layers[l](g_homo, feat, edge_weight, g_homo.ndata[dgl.NTYPE], g_homo.edata[dgl.ETYPE], True)
                    feat = feat.flatten(1)
            new_feat_dict = {}
            for index, ntype in enumerate(g.ntypes):
                new_feat_dict[ntype] = feat[torch.where(g_homo.ndata[dgl.NTYPE] == index)]
        elif mfgs is not None:
            feat = feat_dict
            edge_weight = edge_weight_dict
            for layer, mfg in zip(self.hgn_layers, mfgs):
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


class SimpleHGNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, num_heads, num_etypes, feat_drop=0.0,
                 negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super(SimpleHGNConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.num_etypes = num_etypes

        self.edge_emb = torch.nn.Parameter(torch.empty(size=(self.num_etypes, self.edge_dim)))
        self.W_r = dgl.nn.pytorch.TypedLinear(self.edge_dim, self.edge_dim * self.num_heads, self.num_etypes)
        self.W = torch.nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim * self.num_heads))

        self.a_l = torch.nn.Parameter(torch.empty(size=(1, self.num_heads, self.output_dim)))
        self.a_r = torch.nn.Parameter(torch.empty(size=(1, self.num_heads, self.output_dim)))
        self.a_e = torch.nn.Parameter(torch.empty(size=(1, self.num_heads, self.edge_dim)))

        torch.nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        torch.nn.init.xavier_uniform_(self.W, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = torch.nn.Dropout(feat_drop)
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = torch.nn.Linear(self.input_dim, self.output_dim * num_heads)
        else:
            self.register_buffer("residual", None)

        self.beta = beta

    def forward(self, g_homo, feat, edge_weight, ntype_idxs, etype_idxs, presorted=False):
        emb = self.feat_drop(feat)
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.output_dim)
        emb[torch.isnan(emb)] = 0.0      # num_nodes, num_heads, output_dim

        edge_emb = self.W_r(self.edge_emb[etype_idxs], etype_idxs, presorted).view(-1, self.num_heads, self.edge_dim)     # num_edges, num_heads, edge_dim

        row = g_homo.edges()[0]      # num_edges
        col = g_homo.edges()[1]      # num_edges

        h_l = (self.a_l * emb).sum(dim=-1)[row.long()]     # num_edges, num_heads
        h_r = (self.a_r * emb).sum(dim=-1)[col.long()]     # num_edges, num_heads
        h_e = (self.a_e * edge_emb).sum(dim=-1)     # num_edges, num_heads

        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        # edge_attention = dgl.nn.functional.edge_softmax(g_homo, edge_attention)      # num_edges, num_heads
        edge_attention = torch.sigmoid(edge_attention)      # num_edges, num_heads

        if edge_weight is not None:
            edge_weight = edge_weight.unsqueeze(-1)     # num_edges, 1
            edge_attention = edge_attention * edge_weight       # num_edges, num_heads

        if "alpha" in g_homo.edata.keys():
            res_attn = g_homo.edata["alpha"]
            edge_attention = edge_attention * (1 - self.beta) + res_attn * self.beta

        if self.num_heads == 1:
            edge_attention = edge_attention[:, 0]
            edge_attention = edge_attention.unsqueeze(1)    # num_edges, 1

        with g_homo.local_scope():
            emb = emb.permute(0, 2, 1).contiguous()
            g_homo.edata["alpha"] = edge_attention       # num_edges, num_heads
            g_homo.srcdata["emb"] = emb                  # num_nodes, output_dim, num_heads
            g_homo.update_all(fn.u_mul_e("emb", "alpha", "m"), fn.sum("m", "emb"))
            feat_output = g_homo.dstdata["emb"].view(-1, self.output_dim * self.num_heads)      # num_nodes, output_dim * num_heads

        g_homo.edata["alpha"] = edge_attention
        if g_homo.is_block:
            feat = feat[:g_homo.num_dst_nodes()]
        if self.residual:
            res = self.residual(feat)
            feat_output += res
        if self.activation is not None:
            feat_output = self.activation(feat_output)

        return feat_output

