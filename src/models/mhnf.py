#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F


class MHNF(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_meta_rels, num_meta_paths, num_layers, identity):
        super(MHNF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_meta_rels = num_meta_rels + 1 if identity else num_meta_rels
        self.num_meta_paths = num_meta_paths
        self.num_layers = num_layers
        self.identity = identity

        self.hsaf = HSAF(self.input_dim, self.hidden_dim, self.num_meta_rels, self.num_meta_paths, self.num_layers)
        self.linear = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.nids_of_category = None

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
            h_out = self.hsaf(graph_list, h)
            y = self.linear(h_out)
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

            sg = dgl.graph((nids_src_of_target_etype[mask], nids_dst_of_target_etype[mask]),
                           num_nodes=g_homo.num_nodes())
            sg.edata["w"] = g.edata["edge_weight"][meta_rel]
            graph_list.append(sg)
        if identity:
            x = torch.arange(0, g_homo.num_nodes(), device=device, dtype=torch.int32)
            sg = dgl.graph((x, x))
            sg.edata["w"] = torch.ones(g_homo.num_nodes(), device=device)
            graph_list.append(sg)
        return graph_list, g_homo.ndata["h"], nids_of_category


class HSAF(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_meta_rels, num_meta_paths, num_layers):
        super(HSAF, self).__init__()
        self.num_meta_paths = num_meta_paths
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hlhia_layer = HLHIA(self.input_dim, self.hidden_dim, num_meta_rels, self.num_meta_paths, self.num_layers)
        self.meta_path_attention = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 1),
            torch.nn.Tanh(),
            torch.nn.Linear(1, 1, bias=False),
            torch.nn.ReLU()
        )
        self.layers_attention = torch.nn.ModuleList()
        for i in range(self.num_meta_paths):
            self.layers_attention.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dim, 1),
                    torch.nn.Tanh(),
                    torch.nn.Linear(1, 1, bias=False),
                    torch.nn.ReLU()
                )
            )

    def forward(self, graph_list, feat):
        meta_path_feat_list = self.hlhia_layer(graph_list, feat)
        meta_path_attention_list = []
        for i in range(self.num_meta_paths):
            layer_feature_list = meta_path_feat_list[i]
            layer_attention = self.layers_attention[i]
            for j in range(self.num_layers + 1):
                layer_feat = layer_feature_list[j]
                if j == 0:
                    layer_alpha = layer_attention(layer_feat)
                else:
                    layer_alpha = torch.cat([layer_alpha, layer_attention(layer_feat)], dim=-1)
            layer_beta = torch.softmax(layer_alpha, dim=-1)
            meta_path_attention_list.append(
                torch.bmm(torch.stack(layer_feature_list, dim=-1), layer_beta.unsqueeze(-1)).squeeze(-1)
            )

        for i in range(self.num_meta_paths):
            meta_path_feat = meta_path_attention_list[i]
            if i == 0:
                meta_path_alpha = self.meta_path_attention(meta_path_feat)
            else:
                meta_path_alpha = torch.cat([meta_path_alpha, self.meta_path_attention(meta_path_feat)], dim=-1)
        meta_path_beta = torch.softmax(meta_path_alpha, dim=-1)
        meta_path_attention = torch.bmm(torch.stack(meta_path_attention_list, dim=-1), meta_path_beta.unsqueeze(-1)).squeeze(-1)
        return meta_path_attention


class HLHIA(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_meta_rels, num_meta_paths, num_layers):
        super(HLHIA, self).__init__()
        self.num_meta_paths = num_meta_paths
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(HMAELayer(num_meta_rels, self.num_meta_paths, first=True))
            else:
                self.layers.append(HMAELayer(num_meta_rels, self.num_meta_paths, first=False))

        self.gcn_list = torch.nn.ModuleList()
        for i in range(self.num_meta_paths):
            self.gcn_list.append(dgl.nn.GraphConv(self.input_dim, self.hidden_dim, norm="none", activation=F.relu,
                                                  allow_zero_in_degree=True))
        self.norm = dgl.nn.EdgeWeightNorm(norm="right", eps=1e-6)

    def forward(self, graph_list, feat):
        layer_list = []
        for i in range(len(self.layers)):
            if i == 0:
                new_graph_list, W, first_graph_list = self.layers[i](graph_list)
                layer_list.append(first_graph_list)
                layer_list.append(new_graph_list)
            else:
                new_graph_list, W, first_graph_list = self.layers[i](graph_list, new_graph_list)
                layer_list.append(new_graph_list)

        meta_path_feat_list = []
        for i in range(self.num_meta_paths):
            gcn = self.gcn_list[i]
            layer_feat_list = []
            for j in range(len(layer_list)):
                sg = layer_list[j][i]
                sg = dgl.remove_self_loop(sg)
                edge_weight = sg.edata["w_sum"]
                # cause cuda error because number of edges is larger than number of node pairs
                # sg = dgl.add_self_loop(sg)
                # edge_weight = torch.cat([edge_weight, torch.full((sg.num_nodes(),), 1, device=sg.device)])
                edge_weight = self.norm(sg, edge_weight)
                sg = dgl.graph((sg.edges()[1], sg.edges()[0]), num_nodes=sg.num_nodes(), device=sg.device)  # switch src and dst nodes
                layer_feat_list.append(gcn(sg, feat, edge_weight=edge_weight))
            meta_path_feat_list.append(layer_feat_list)
        return meta_path_feat_list


class HMAELayer(torch.nn.Module):
    def __init__(self, num_meta_rels, num_meta_paths, first=True):
        super(HMAELayer, self).__init__()
        self.num_meta_rels = num_meta_rels
        self.num_meta_paths = num_meta_paths
        self.first = first

        self.norm = dgl.nn.EdgeWeightNorm(norm="right", eps=1e-6)
        if self.first:
            self.conv1 = GTConv(self.num_meta_rels, self.num_meta_paths, softmax_flag=False)
            self.conv2 = GTConv(self.num_meta_rels, self.num_meta_paths, softmax_flag=False)
        else:
            self.conv1 = GTConv(self.num_meta_rels, self.num_meta_paths, softmax_flag=False)

    def softmax_norm(self, graph_list):
        norm_graph_list = []
        for g in graph_list:
            g.edata["w_sum"] = self.norm(g, torch.exp(g.edata["w_sum"]))
            norm_graph_list.append(g)
        return norm_graph_list

    def forward(self, graph_list, H_=None):
        if self.first:
            result_A = self.softmax_norm(self.conv1(graph_list))
            result_B = self.softmax_norm(self.conv2(graph_list))
            W = [self.conv1.weight.detach(), self.conv2.weight.detach()]
        else:
            result_A = H_
            result_B = self.conv1(graph_list)
            W = [self.conv1.weight.detach()]
        new_graph_list = []
        for i in range(len(result_A)):
            g = dgl.adj_product_graph(result_A[i], result_B[i], "w_sum")
            new_graph_list.append(g)
        first_graph_list = result_A
        return new_graph_list, W, first_graph_list


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