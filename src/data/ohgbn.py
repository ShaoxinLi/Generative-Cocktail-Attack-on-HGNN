#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import dgl
import torch


class OHGBDataset(dgl.data.DGLDataset):

    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {}

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        assert name in ["ohgbl-MTWM", "ohgbl-yelp1", "ohgbl-yelp2", "ohgbl-Freebase",
                        "ohgbn-Freebase", "ohgbn-yelp2", "ohgbn-acm", "ohgbn-imdb"]
        # self.data_path = "./openhgnn/dataset/{}.zip".format(name)
        # self.g_path = "./openhgnn/dataset/{}/graph.bin".format(name)
        # raw_dir = "./openhgnn/dataset"
        self.data_path = "/home/share/Datasets/openhgnn/dataset/{}.zip".format(name)
        self.g_path = "/home/share/Datasets/openhgnn/dataset/{}/graph.bin".format(name)
        raw_dir = "/home/share/Datasets/openhgnn/dataset"
        url = self._prefix + "dataset/{}.zip".format(name)
        super(OHGBDataset, self).__init__(name=name, url=url, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    def download(self):
        # download raw data to local disk
        # path to store the file
        if os.path.exists(self.data_path):  # pragma: no cover
           pass
        else:
            file_path = os.path.join(self.raw_dir)
            # download file
            dgl.data.utils.download(self.url, path=file_path)
        dgl.data.utils.extract_archive(self.data_path, os.path.join(self.raw_dir, self.name))

    def process(self):
        # process raw data to graphs, labels, splitting masks
        g, _ = dgl.data.utils.load_graphs(self.g_path)
        self._g = g[0]

    def __getitem__(self, idx):
        # get one example by index
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        # number of data examples
        return 1

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass


def load_ohgbn(name, device):
    if name == "acm":
        dataset = OHGBDataset("ohgbn-acm")
    elif name == "imdb":
        dataset = OHGBDataset("ohgbn-imdb")
    else:
        assert False, f"Dataset {name} is not supported yet."

    g = dataset[0]
    g = g.int()
    g = g.to(device, non_blocking=True)

    if name == "acm":
        g.category = "paper"
    elif name == "imdb":
        g.category = "movie"
    else:
        assert False
    g.num_classes = len(torch.unique(g.nodes[g.category].data["label"]))
    if name == "imdb":
        g.num_classes = 3
    g.train_idxs = torch.nonzero(g.nodes[g.category].data["train_mask"], as_tuple=False).squeeze().int()
    if "valid_mask" in g.nodes[g.category].data.keys():
        g.val_idxs = torch.nonzero(g.nodes[g.category].data["valid_mask"], as_tuple=False).squeeze().int()
    g.test_idxs = torch.nonzero(g.nodes[g.category].data["test_mask"], as_tuple=False).squeeze().int()
    g.labels = g.nodes[g.category].data["label"].int()
    return g


