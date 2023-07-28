#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F


class GTN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_meta_rels, num_meta_paths, num_layers, identity):
        super(GTN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_meta_rels = num_meta_rels + 1 if identity else num_meta_rels
        self.num_meta_paths = num_meta_paths
        self.num_layers = num_layers
        self.identity = identity

        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(GTConv(self.num_meta_rels, self.num_meta_paths))
        self.params = torch.nn.ParameterList()
        for i in range(self.num_meta_paths):
            self.params.append(torch.nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim)))
        self.gcn = GCNConv()
        self.norm = dgl.nn.EdgeWeightNorm(norm="right", eps=1e-6)
        self.linear1 = torch.nn.Linear(self.hidden_dim * self.num_meta_paths, self.hidden_dim)
        self.linear2 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.nids_of_category = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.params is not None:
            for para in self.params:
                torch.nn.init.xavier_uniform_(para)

    def forward(self, g, mfgs, feat_dict, edge_weight_dict, out_key):
        assert mfgs is None
        assert "edge_weight" in g.edata.keys()
        assert out_key is not None
        with g.local_scope():
            g.ndata["h"] = feat_dict
            # if self.nids_of_category is None:
            #     self.graph_list, h, self.nids_of_category = self.transform_relation_graph_list(g, out_key, self.identity)
            # else:
            #     g_homo = dgl.to_homogeneous(g, ndata=["h"])
            #     h = g_homo.ndata["h"]
            self.graph_list, h, self.nids_of_category = self.transform_relation_graph_list(g, out_key, self.identity)
            graph_list = self.graph_list
            H = []
            for i in range(self.num_meta_paths):
                H.append(torch.matmul(h, self.params[i]))
            for i in range(self.num_layers):
                graph_list_hat = self.layers[i](graph_list)
                for j in range(self.num_meta_paths):
                    edge_weight = self.norm(graph_list_hat[j], graph_list_hat[j].edata["w_sum"])
                    H[j] = self.gcn(graph_list_hat[j], H[j], edge_weight=edge_weight)

            X_ = self.linear1(torch.cat(H, dim=1))
            X_ = F.relu(X_)
            y = self.linear2(X_)
            return y[self.nids_of_category],

    @staticmethod
    def transform_relation_graph_list(g, category, identity=True):
        assert "edge_weight" in g.edata.keys()
        for i, ntype in enumerate(g.ntypes):
            if ntype == category:
                category_idx = i
        g_homo = dgl.to_homogeneous(g, ndata=["h"], edata=["edge_weight"])
        nids_of_category = (g_homo.ndata[dgl.NTYPE] == category_idx).to("cpu")
        nids_of_category = torch.arange(g_homo.num_nodes())[nids_of_category]

        ntype2idx = {ntype: i for i, ntype in enumerate(g.ntypes)}
        idx2etype = {i: etype for i, etype in enumerate(g.etypes)}

        nids_src, nids_dst = g_homo.edges()
        n_type_ids = g_homo.ndata[dgl.NTYPE]
        e_type_ids = g_homo.edata[dgl.ETYPE]
        device = g_homo.device

        graph_list = []
        for meta_rel in g.canonical_etypes:
            ntype_src, etype, ntype_dst = meta_rel
            e_type_idxs_of_target_etype = [idx for idx, edge_type in idx2etype.items() if edge_type == etype]
            e_type_idxs_of_target_etype = torch.tensor(e_type_idxs_of_target_etype, device=device)
            eids_of_target_etype = torch.isin(e_type_ids, e_type_idxs_of_target_etype)

            nids_src_of_target_etype = nids_src[eids_of_target_etype]
            nids_dst_of_target_etype = nids_dst[eids_of_target_etype]

            n_type_ids_of_src_nodes = n_type_ids[nids_src_of_target_etype.long()]
            n_type_ids_of_dst_nodes = n_type_ids[nids_dst_of_target_etype.long()]

            ntype_src_idx = ntype2idx[ntype_src]
            ntype_dst_idx = ntype2idx[ntype_dst]

            mask_of_src_nodes = torch.isin(n_type_ids_of_src_nodes, torch.tensor([ntype_src_idx], device=device))
            mask_of_dst_nodes = torch.isin(n_type_ids_of_dst_nodes, torch.tensor([ntype_dst_idx], device=device))
            mask = torch.logical_and(mask_of_src_nodes, mask_of_dst_nodes)

            sg = dgl.graph((nids_src_of_target_etype[mask], nids_dst_of_target_etype[mask]), num_nodes=g_homo.num_nodes())
            sg.edata["w"] = g.edata["edge_weight"][meta_rel]
            graph_list.append(sg)
        if identity:
            x = torch.arange(0, g_homo.num_nodes(), device=device, dtype=torch.int32)
            sg = dgl.graph((x, x))
            sg.edata["w"] = torch.ones(g_homo.num_nodes(), device=device)
            graph_list.append(sg)
        return graph_list, g_homo.ndata["h"], nids_of_category


class GCNConv(torch.nn.Module):
    def __init__(self,):
        super(GCNConv, self).__init__()

    def forward(self, g, feat, edge_weight=None):
        g = dgl.graph((g.edges()[1], g.edges()[0]), num_nodes=g.num_nodes(), device=g.device)   # switch src and dst nodes
        with g.local_scope():
            if edge_weight is not None:
                assert edge_weight.shape[0] == g.num_edges()
                g.srcdata["h"] = feat
                g.edata["_edge_weight"] = edge_weight
                g.update_all(fn.u_mul_e("h", "_edge_weight", "m"), fn.sum(msg="m", out="h"))
                rst = g.dstdata["h"]
        return rst


class GTConv(torch.nn.Module):
    def __init__(self, num_meta_rels, num_meta_paths, softmax_flag=True):
        super(GTConv, self).__init__()
        self.num_meta_rels = num_meta_rels
        self.num_meta_paths = num_meta_paths
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_meta_paths, self.num_meta_rels))
        self.softmax_flag = softmax_flag
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=0.01)

    def forward(self, graph_list):
        if self.softmax_flag:
            Filter = F.softmax(self.weight, dim=1)
        else:
            Filter = self.weight
        num_meta_paths = Filter.shape[0]
        results = []

        for i in range(num_meta_paths):
            for j, g in enumerate(graph_list):
                g.edata["w_sum"] = g.edata["w"] * Filter[i][j]
            sum_g = dgl.adj_sum_graph(graph_list, "w_sum")
            results.append(sum_g)
        return results
