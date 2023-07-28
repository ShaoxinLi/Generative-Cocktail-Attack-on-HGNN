#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import dgl.function as fn
import copy
import torch
import torch.nn.functional as F


class HAN(torch.nn.Module):
    def __init__(self, meta_paths, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super(HAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, self.input_dim, self.hidden_dim, num_heads, dropout)
        )
        for l in range(1, self.num_layers):
            self.layers.append(
                HANLayer(meta_paths, self.hidden_dim * num_heads, self.hidden_dim, num_heads, dropout)
            )
        self.predict = torch.nn.Linear(self.hidden_dim * num_heads, self.output_dim)

    def forward(self, g, mfgs, feat_dict, edge_weight_dict, out_key):
        assert mfgs is None
        assert "edge_weight" in g.edata.keys()
        new_feat_dict = copy.deepcopy(feat_dict)
        for han_layer in self.layers:
            feat, ntype_dst = han_layer(g, new_feat_dict)
            new_feat_dict[ntype_dst] = feat
        if out_key is not None:
            return self.predict(new_feat_dict[out_key]),
        else:
            return new_feat_dict,


class HANLayer(torch.nn.Module):
    def __init__(self, meta_paths, input_dim, output_dim, num_heads, dropout):
        super(HANLayer, self).__init__()
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self.gat_layers = torch.nn.ModuleList()
        for i in range(len(self.meta_paths)):
            self.gat_layers.append(
                GATConv(
                    input_dim, output_dim, num_heads, dropout, dropout,
                    activation=F.elu, allow_zero_in_degree=True
                )
            )
        self.semantic_attention = SemanticAttention(
            input_dim=output_dim * num_heads, output_dim=128
        )

    def forward(self, g, feat_dict):
        semantic_embeddings = []
        for i, meta_path in enumerate(self.meta_paths):
            meta_path_g = self._get_meta_path_based_graph(g, meta_path)
            ntype_src, ntype_dst = meta_path[0][0], meta_path[-1][-1]
            feat_src = feat_dict[ntype_src]
            feat_dst = feat_dict[ntype_dst]
            feat = self.gat_layers[i](meta_path_g, (feat_src, feat_dst)).flatten(1)        # num_dst_nodes, output_dim * num_heads
            semantic_embeddings.append(feat)
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)   # num_dst_nodes, num_meta_paths, output_dim * num_heads
        return self.semantic_attention(semantic_embeddings), ntype_dst     # num_dst_nodes, output_dim * num_heads

    @staticmethod
    def _get_meta_path_based_graph(g, meta_path):
        # meta_path is list of canonical etypes
        assert len(meta_path) >= 2
        g_a, g_b = g[meta_path[0]], g[meta_path[1]]
        g_a = dgl.adj_product_graph(g_a, g_b, "edge_weight")
        if len(meta_path) > 2:
            for i in range(2, len(meta_path)):
                g_b = g[meta_path[i]]
                g_a = dgl.adj_product_graph(g_a, g_b, "edge_weight")
        return g_a


class SemanticAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SemanticAttention, self).__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(output_dim, 1, bias=False)
        )

    def forward(self, z):       # num_nodes, num_meta_paths, input_dim
        w = self.project(z).mean(0)     # num_meta_paths, 1
        beta = torch.softmax(w, dim=0)  # num_meta_paths, 1
        beta = beta.expand((z.shape[0],) + beta.shape)  # num_nodes, num_meta_paths, 1
        return (beta * z).sum(1)        # num_nodes, input_dim


class GATConv(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = dgl.utils.expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = torch.nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = torch.nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = torch.nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = torch.nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = torch.nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = torch.nn.Dropout(feat_drop)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer("bias", None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = torch.nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = dgl.nn.pytorch.utils.Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            torch.nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            torch.nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            torch.nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.attn_l, gain=gain)
        torch.nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, torch.nn.Linear):
            torch.nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise dgl._ffi.base.DGLError("There are 0-in-degree nodes in the graph, "
                                                 "output for those nodes will be invalid. "
                                                 "This is harmful for some applications, "
                                                 "causing silent performance regression. "
                                                 "Adding self-loop on the input graph by "
                                                 "calling `g = dgl.add_self_loop(g)` will resolve "
                                                 "the issue. Setting ``allow_zero_in_degree`` "
                                                 "to be `True` when constructing this module will "
                                                 "suppress the check and let the code run.")

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            # graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            a = self.attn_drop(torch.sigmoid(e))        # num_edges, num_heads, 1

            if "edge_weight" in graph.edata.keys():
                edge_weight = graph.edata["edge_weight"]
                edge_weight = edge_weight.view(-1, 1, 1)        # num_edges, 1, 1
                a = a * edge_weight

            graph.edata["a"] = a

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst
