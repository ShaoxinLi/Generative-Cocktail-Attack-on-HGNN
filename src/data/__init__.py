#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .rdf import load_rdf
from .ohgbn import load_ohgbn
from .dblp import load_dblp
from .dblp import load_dblp_paper


def get_dataset(dataset_name, device):

    if dataset_name in ["aifb", "mutag", "bgs", "am"]:
        g = load_rdf(dataset_name, device=device)
    elif dataset_name in ["acm", "imdb"]:
        g = load_ohgbn(dataset_name, device=device)
    elif dataset_name in ["dblp"]:
        g = load_dblp(device=device)
    elif dataset_name in ["dblp_paper"]:
        g = load_dblp_paper(device=device)
    else:
        assert False, f"Dataset {dataset_name} is not supported yet."
    return g