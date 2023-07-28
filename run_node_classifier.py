#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import argparse

from src.data import get_dataset
from src.models import get_network, get_num_parameters, get_num_trainable_parameters
from src.metrics import Accuracy, MacroF1Score, MicroF1Score
from src.solvers.callbacks import *
from src.solvers import NodeClassifier
from src.utils import CheckpointIO, setup_cfg, print_cfg


def parse_arguments():

    def str_or_int_or_none(value):
        if value == "":
            return None
        elif value.isdigit():
            return int(value)
        return value

    def list_or_none(value):
        if value == "":
            return None
        else:
            return [int(i) for i in value.split(",")]

    def true_or_false(value):
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            assert False

    parser = argparse.ArgumentParser()

    # Experiment args
    parser.add_argument("--exp_root_dir", type=str, default="./archive", help="The root dir for storing results")
    parser.add_argument("--dir_suffix", type=str, default="", help="The suffix of the result directory name")
    parser.add_argument("--device", type=str_or_int_or_none, default=None, help="Device for computing (default: None)")
    parser.add_argument("--seed", type=str_or_int_or_none, default=3407, help="Random seed (default: 3407)")

    # Network args
    parser.add_argument("--net_arch", type=str, default="rgcn", choices=["rgcn", "hgt", "simple_hgn", "hetsann", "han", "gtn", "mhnf"], help="Architecture for the classifier (default: rgcn)")
    parser.add_argument("--net_ckpt_path", type=str_or_int_or_none, default=None, help="The checkpoint file for the classifier (default: '')")
    parser.add_argument("--hidden_dim", type=int, default=16, help="The dimension of hidden layer (default: '16')")

    # Dataset args
    parser.add_argument("--dataset", type=str, default="acm", choices=["acm", "imdb", "dblp", "dblp_paper"], help="Used dataset (default: acm)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--validation_split", type=float, default=-1, help="Percentage used to split the validation set (default: -1)")

    # Trainer args
    parser.add_argument("--full_batch", type=true_or_false, default=True)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train (default: 100)")
    parser.add_argument("--patience", type=int, default=-1, help="The patience in the earlystopping strategy (default: -1)")
    parser.add_argument("--print_freq", type=int, default=1, help="Frequency of printing training logs (default: 1)")
    parser.add_argument("--auto_restore", type=true_or_false, default=True)
    parser.add_argument("--verbose", type=true_or_false, default=False)

    # Optimizing args
    parser.add_argument("--opt_alg", type=str, default="adam", choices=["sgd", "adam"], help="Used optimizer (default: adam)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate (default: 1e-2)")

    args = parser.parse_args()
    return args


def prepare_graph(args):

    # get the graph
    args.logger.info(f"{'=' * 20}> Loading dataset {args.dataset}:")
    g = get_dataset(dataset_name=args.dataset, device=args.device)

    # split the training idxs into training and validating parts
    if not hasattr(g, "val_idxs"):
        if 0.0 < args.validation_split < 1.0:
            val_idxs = g.train_idxs[:int(len(g.train_idxs) * args.validation_split)]
            train_idxs = g.train_idxs[int(len(g.train_idxs) * args.validation_split):]
        else:
            val_idxs = train_idxs = g.train_idxs
        g.train_idxs = train_idxs
        g.val_idxs = val_idxs
    args.logger.info(f"# Training nodes: {len(g.train_idxs)}")
    args.logger.info(f"# Validating nodes: {len(g.val_idxs)}")
    args.logger.info(f"# Testing nodes: {len(g.test_idxs)}")

    # for stochastic training
    if not args.full_batch:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        train_loader = dgl.dataloading.DataLoader(
            graph=g, indices={g.category: g.train_idxs}, graph_sampler=sampler,
            device=args.device, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = dgl.dataloading.DataLoader(
            graph=g, indices={g.category: g.val_idxs}, graph_sampler=sampler,
            device=args.device, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        test_loader = dgl.dataloading.DataLoader(
            graph=g, indices={g.category: g.test_idxs}, graph_sampler=sampler,
            device=args.device, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        args.train_loader = train_loader
        args.val_loader = val_loader
        args.test_loader = test_loader
    else:
        args.train_loader = None
        args.val_loader = None
        args.test_loader = None
    args.g = g

    # for meta-path based HGNNs we need to add edge weights
    if args.net_arch in ["han", "gtn", "mhnf"]:
        edge_weights_dict = {}
        for meta_rel in args.g.canonical_etypes:
            num_edges = args.g.num_edges(meta_rel)
            edge_weights_dict[meta_rel] = torch.ones(num_edges, device=args.device)
        args.g.edata["edge_weight"] = edge_weights_dict


def prepare_model(args):

    args.logger.info(f"{'=' * 20}> Loading network: {args.net_arch}")
    if "h" not in args.g.nodes[args.g.category].data:
        if "feat" in args.g.ndata.keys():  # dblp
            input_dim = 334
        else:
            input_dim = args.hidden_dim
    else:
        input_dim = args.g.nodes[args.g.category].data["h"].size(-1)
    net = get_network(
        net_arch=args.net_arch, input_dim=input_dim, hidden_dim=args.hidden_dim,
        output_dim=args.g.num_classes, meta_rels=args.g.canonical_etypes, dataset_name=args.dataset
    )
    args.net = net.to(args.device, non_blocking=True)

    # initialize from checkpoint
    if args.net_ckpt_path is not None:
        assert not args.auto_restore
        ckptio = CheckpointIO(ckpt_dir=args.exp_dir, device=args.device, net_state=args.net)
        ckptio.load_from_path(args.net_ckpt_path)
    if args.verbose:
        args.logger.info(f"{'=' * 20}> Network :\n {args.net}")
        args.logger.info(f"Total # parameters: {get_num_parameters(args.net)}")
        args.logger.info(f"# Trainable parameters: {get_num_trainable_parameters(args.net)}")


if __name__ == "__main__":

    args = parse_arguments()
    args.exp_type = "node_classification"
    setup_cfg(args)
    print_cfg(args)

    # instantiate an entity classifier
    classifier = NodeClassifier(
        opt_alg=args.opt_alg, lr=args.lr, num_epochs=args.num_epochs,
        device=args.device, seed=args.seed
    )

    # preparation
    prepare_graph(args)
    prepare_model(args)

    # train
    callbacks = []
    callbacks.append(MetricTracker({"acc": Accuracy(), "macro-f1": MacroF1Score(), "micro-f1": MicroF1Score()}))
    callbacks.append(FileLogger(args.logger, args.num_epochs, args.verbose, args.print_freq))
    callbacks.append(Recorder(args.exp_dir))
    callbacks.append(CheckpointSaver(args.exp_dir, monitor="loss"))
    if args.patience > 0:
        callbacks.append(EarlyStopper(monitor="loss", patience=args.patience))
    callbacks.append(MetricPlotter(args.exp_dir))
    classifier.fit(
        net=args.net, g=args.g, full_batch=args.full_batch,
        train_loader=args.train_loader, val_loader=args.val_loader, callbacks=callbacks
    )

    # test
    callbacks = []
    callbacks.append(MetricTracker({"acc": Accuracy(), "macro-f1": MacroF1Score(), "micro-f1": MicroF1Score()}))
    test_losses, test_metrics = classifier.test(
        net=args.net, g=args.g, full_batch=args.full_batch, test_loader=args.test_loader,
        callbacks=callbacks, exp_dir=args.exp_dir, auto_restore=args.auto_restore
    )
    for k, v in test_losses.items():
        args.logger.info(f"Test {k}: {v:.3f}")
    for k, v in test_metrics.items():
        args.logger.info(f"Test {k}: {v:.3f}")