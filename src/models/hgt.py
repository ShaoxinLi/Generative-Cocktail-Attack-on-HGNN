#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import dgl.function as fn
import math
import torch
import torch.nn.functional as F


class HGT(torch.nn.Module):
    def __init__(self, meta_rels, input_dim, hidden_dim, output_dim, num_layers, num_heads, use_norm=True):
        super(HGT, self).__init__()
        ntypes_src = [meta_rel[0] for meta_rel in meta_rels]
        ntypes_dst = [meta_rel[-1] for meta_rel in meta_rels]
        ntypes = list(set(ntypes_src + ntypes_dst))
        etypes = list(set([meta_rel[1] for meta_rel in meta_rels]))
        ntypes.sort()
        etypes.sort()
        self.node_dict = {ntype: i for i, ntype in enumerate(ntypes)}
        self.edge_dict = {etype: i for i, etype in enumerate(etypes)}
        self.gcs = torch.nn.ModuleList()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.adapt_ws = torch.nn.ModuleList()
        for t in range(len(self.node_dict)):
            self.adapt_ws.append(torch.nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.num_layers):
            self.gcs.append(
                HGTLayer(hidden_dim, hidden_dim, meta_rels, self.node_dict, self.edge_dict, num_heads, use_norm=use_norm)
            )
        self.out = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, g, mfgs, feat_dict, edge_weight_dict, out_key):
        assert mfgs is None
        new_feat_dict = {}
        for ntype in g.ntypes:
            n_id = self.node_dict[ntype]
            new_feat_dict[ntype] = F.gelu(self.adapt_ws[n_id](feat_dict[ntype]))
        for i in range(self.num_layers):
            new_feat_dict = self.gcs[i](g, new_feat_dict, edge_weight_dict)
        if out_key is not None:
            return self.out(new_feat_dict[out_key]),
        else:
            return new_feat_dict,


class HGTLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, meta_rels, node_dict, edge_dict, num_heads, dropout=0.2, use_norm=False):
        super(HGTLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.meta_rels = meta_rels
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_ntypes = len(node_dict)
        self.num_etypes = len(edge_dict)
        self.num_heads = num_heads
        self.d_k = self.output_dim // self.num_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = torch.nn.ModuleList()
        self.q_linears = torch.nn.ModuleList()
        self.m_linears = torch.nn.ModuleList()
        self.a_linears = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_ntypes):
            self.k_linears.append(torch.nn.Linear(input_dim, output_dim))
            self.q_linears.append(torch.nn.Linear(input_dim, output_dim))
            self.m_linears.append(torch.nn.Linear(input_dim, output_dim))
            self.a_linears.append(torch.nn.Linear(output_dim, output_dim))
            if self.use_norm:
                self.norms.append(torch.nn.LayerNorm(output_dim))

        self.w_pri = torch.nn.Parameter(torch.ones(self.num_etypes, self.num_heads))
        self.w_att = torch.nn.Parameter(torch.Tensor(self.num_etypes, self.num_heads, self.d_k, self.d_k))
        self.w_msg = torch.nn.Parameter(torch.Tensor(self.num_etypes, self.num_heads, self.d_k, self.d_k))
        self.skip = torch.nn.Parameter(torch.ones(self.num_etypes))
        self.drop = torch.nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.w_att)
        torch.nn.init.xavier_uniform_(self.w_msg)

    def forward(self, g, feat_dict, edge_weight_dict):
        with g.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for meta_rel in g.canonical_etypes:
                ntype_src, etype, ntype_dst = meta_rel
                meta_rel_g = g[ntype_src, etype, ntype_dst]

                k_linear = self.k_linears[node_dict[ntype_src]]
                q_linear = self.q_linears[node_dict[ntype_dst]]
                m_linear = self.m_linears[node_dict[ntype_src]]

                # self-transform
                k = k_linear(feat_dict[ntype_src]).view(-1, self.num_heads, self.d_k)   # num_nodes, num_heads, d_k
                q = q_linear(feat_dict[ntype_dst]).view(-1, self.num_heads, self.d_k)   # num_nodes, num_heads, d_k
                m = m_linear(feat_dict[ntype_src]).view(-1, self.num_heads, self.d_k)   # num_nodes, num_heads, d_k

                # get edge-type specific weights
                e_id = edge_dict[etype]
                w_pri = self.w_pri[e_id]      # num_heads
                w_att = self.w_att[e_id]      # num_heads, d_k, d_k
                w_msg = self.w_msg[e_id]      # num_heads, d_k, d_k

                # src features x edge-type specific weights
                k = torch.einsum("bij,ijk->bik", k, w_att)   # num_nodes, num_heads, d_k
                m = torch.einsum("bij,ijk->bik", m, w_msg)   # num_nodes, num_heads, d_k

                meta_rel_g.srcdata["k"] = k
                meta_rel_g.dstdata["q"] = q
                meta_rel_g.srcdata["m_%d" % e_id] = m

                # compute attention scores
                meta_rel_g.apply_edges(fn.v_dot_u("q", "k", "t"))   # num_edges, num_heads
                attn_score = (meta_rel_g.edata.pop("t").sum(-1) * w_pri / self.sqrt_dk)  # num_edges, num_heads

                # softmax the attention scores
                # attn_score = dgl.nn.functional.edge_softmax(meta_rel_g, attn_score, norm_by="dst")  # num_edges, num_heads
                attn_score = torch.sigmoid(attn_score)      # num_edges, num_heads

                if edge_weight_dict is not None:
                    edge_weights = edge_weight_dict[meta_rel].unsqueeze(-1)       # num_edges, 1
                    attn_score = attn_score * edge_weights      # num_edges, num_heads

                attn_score = attn_score.unsqueeze(-1)   # num_edges, num_heads, 1
                meta_rel_g.edata["t"] = attn_score

            g.multi_update_all(
                etype_dict={meta_rel: (fn.u_mul_e("m_%d" % edge_dict[meta_rel[1]], "t", "m"), fn.sum("m", "t"),) for meta_rel in self.meta_rels},
                cross_reducer="mean"
            )
            # g.multi_update_all(
            #     etype_dict={etype: (fn.u_mul_e("m_%d" % e_id, "t", "m"), fn.sum("m", "t"),) for etype, e_id in edge_dict.items()},
            #     cross_reducer="mean"
            # )

            new_feat_dict = {}
            for ntype in g.ntypes:
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = g.nodes[ntype].data["t"].view(-1, self.output_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + feat_dict[ntype] * (1 - alpha)
                if self.use_norm:
                    new_feat_dict[ntype] = self.norms[n_id](trans_out)
                else:
                    new_feat_dict[ntype] = trans_out
            return new_feat_dict
