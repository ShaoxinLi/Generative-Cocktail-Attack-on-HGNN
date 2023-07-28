#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import torch


def load_rdf(name, device):
    if name == "aifb":
        dataset = dgl.data.rdf.AIFBDataset()
    elif name == "mutag":
        dataset = dgl.data.rdf.MUTAGDataset()
    elif name == "bgs":
        dataset = dgl.data.rdf.BGSDataset()
    elif name == "am":
        dataset = dgl.data.rdf.AMDataset()
    else:
        assert False, f"Dataset {name} is not supported yet."

    g = dataset[0]
    g = g.int()
    g = g.to(device, non_blocking=True)

    g.category = dataset.predict_category
    g.num_classes = dataset.num_classes
    g.train_idxs = torch.nonzero(g.nodes[g.category].data["train_mask"], as_tuple=False).squeeze().int()
    g.test_idxs = torch.nonzero(g.nodes[g.category].data["test_mask"], as_tuple=False).squeeze().int()
    g.labels = g.nodes[g.category].data["labels"].int()
    return g

