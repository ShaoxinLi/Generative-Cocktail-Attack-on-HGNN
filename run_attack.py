#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from src.data import get_dataset
from src.models import get_network, get_num_parameters, get_num_trainable_parameters, set_parameter_requires_grad
from src.models import Predictor
from src.metrics import FoolingRatio, MacroF1Score, MicroF1Score
from src.solvers.callbacks import *
from src.solvers import Attack
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
    parser.add_argument("--net_arch", type=str, default="rgcn", choices=["rgcn"], help="Architecture for the classifier (default: rgcn)")
    parser.add_argument("--net_hidden_dim", type=int, default=128, help="The dimension of hidden layer (default: '16')")
    parser.add_argument("--net_output_dim", type=int, default=64, help="The dimension of hidden layer (default: '16')")
    parser.add_argument("--net_ckpt_path", type=str_or_int_or_none, default=None)
    parser.add_argument("--target_net_arch", type=str, default="rgcn", choices=["rgcn", "hgt", "simple_hgn", "hetsann", "han", "gtn", "mhnf"])
    parser.add_argument("--target_net_hidden_dim", type=int, default=16, help="The dimension of hidden layer (default: '16')")
    parser.add_argument("--target_net_ckpt_path", type=str_or_int_or_none, default=None, help="The checkpoint file for the classifier (default: '')")

    # Dataset args
    parser.add_argument("--dataset", type=str, default="acm", choices=["acm", "imdb", "dblp", "dblp_paper"], help="Used dataset (default: acm)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--validation_split", type=float, default=-1, help="Percentage used to split the validation set (default: -1)")
    parser.add_argument("--search_scope", type=int, default=0)
    parser.add_argument("--num_samplings", type=int, default=-1)

    # Trainer args
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train (default: 100)")
    parser.add_argument("--xi", type=int, default=5)
    parser.add_argument("--patience", type=int, default=-1, help="The patience in the earlystopping strategy (default: -1)")
    parser.add_argument("--print_freq", type=int, default=1, help="Frequency of printing training logs (default: 1)")
    parser.add_argument("--auto_restore", type=true_or_false, default=True)
    parser.add_argument("--verbose", type=true_or_false, default=False)
    parser.add_argument("--lmd", type=float, default=1.0)

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
    args.g = g


def prepare_models(args):

    # load target net
    args.logger.info(f"{'=' * 20}> Loading target network: {args.target_net_arch}")
    if "h" not in args.g.ndata.keys():
        if "feat" in args.g.ndata.keys():  # dblp
            input_dim = 334
        else:
            input_dim = args.target_net_hidden_dim
    else:
        input_dim = args.g.nodes[args.g.category].data["h"].size(-1)
    args.target_net = get_network(
        net_arch=args.target_net_arch, input_dim=input_dim, hidden_dim=args.target_net_hidden_dim,
        output_dim=args.g.num_classes, meta_rels=args.g.canonical_etypes, dataset_name=args.dataset
    )
    args.target_net = args.target_net.to(args.device, non_blocking=True)

    assert args.target_net_ckpt_path is not None
    if "h" not in args.g.ndata.keys():
        from src.models.rgcn import RelGraphEmbed
        ntype2num = {ntype: args.g.num_nodes(ntype) for ntype in args.g.ntypes}
        if "feat" in args.g.ndata.keys():  # dblp
            args.embeds = RelGraphEmbed(ntype2num, 334)
        else:
            args.embeds = RelGraphEmbed(ntype2num, args.target_net_hidden_dim)
        args.embeds = args.embeds.to(args.device, non_blocking=True)
        ckptio = CheckpointIO(ckpt_dir=args.exp_dir, device=args.device, net_state=args.target_net, embed_state=args.embeds)
        ckptio.load_from_path(args.target_net_ckpt_path)
        set_parameter_requires_grad(args.target_net, requires_grad=False)
        set_parameter_requires_grad(args.embeds, requires_grad=False)
    else:
        args.embeds = None
        ckptio = CheckpointIO(ckpt_dir=args.exp_dir, device=args.device, net_state=args.target_net)
        ckptio.load_from_path(args.target_net_ckpt_path)
        set_parameter_requires_grad(args.target_net, requires_grad=False)
    if args.verbose:
        args.logger.info(f"{'=' * 20}> Target network :\n {args.target_net}")
        args.logger.info(f"Total # parameters: {get_num_parameters(args.target_net)}")
        args.logger.info(f"# Trainable parameters: {get_num_trainable_parameters(args.target_net)}")

    # prepare the attack net
    args.logger.info(f"{'=' * 20}> Loading attack network: {args.net_arch}")
    args.net = get_network(
        net_arch=args.net_arch, input_dim=input_dim, hidden_dim=args.net_hidden_dim,
        output_dim=args.net_output_dim, meta_rels=args.g.canonical_etypes, dataset_name=args.dataset
    )
    args.net = args.net.to(args.device, non_blocking=True)
    args.predictor = Predictor(args.net_output_dim)
    args.predictor = args.predictor.to(args.device, non_blocking=True)
    args.test_slope = None
    if args.net_ckpt_path is not None:
        assert not args.auto_restore
        ckptio = CheckpointIO(
            ckpt_dir=args.exp_dir, device=args.device, net_state=args.net,
            predictor_state=args.predictor, slope=args.test_slope
        )
        module_dict = ckptio.load_from_path(args.net_ckpt_path)
        args.test_slope = module_dict["slope"]
        args.logger.info(f"Test slope: {args.test_slope}")
    if args.verbose:
        args.logger.info(f"{'=' * 20}> Attack network :\n {args.net}")
        args.logger.info(f"Total # parameters: {get_num_parameters(args.net) + get_num_parameters(args.predictor)}")
        args.logger.info(f"# Trainable parameters: {get_num_trainable_parameters(args.net) + get_num_parameters(args.predictor)}")


if __name__ == "__main__":

    args = parse_arguments()
    args.exp_type = "attack"
    setup_cfg(args)
    print_cfg(args)

    # preparation
    prepare_graph(args)
    prepare_models(args)

    # instantiate an attacker
    attacker = Attack(
        opt_alg=args.opt_alg, lr=args.lr, num_epochs=args.num_epochs,
        device=args.device, seed=args.seed
    )

    # attack
    callbacks = []
    callbacks.append(MetricTracker({"fr": FoolingRatio(), "macro-f1": MacroF1Score(), "micro-f1": MicroF1Score()}))
    callbacks.append(FileLogger(args.logger, args.num_epochs, args.verbose, args.print_freq))
    callbacks.append(Recorder(args.exp_dir))
    callbacks.append(CheckpointSaver(args.exp_dir, monitor="attack_loss"))
    if args.patience > 0:
        callbacks.append(EarlyStopper(monitor="attack_loss", patience=args.patience))
    callbacks.append(MetricPlotter(args.exp_dir))
    attacker.attack(
        g=args.g, net=args.net, predictor=args.predictor, target_net=args.target_net, embeds=args.embeds,
        xi=args.xi, search_scope=args.search_scope, num_samplings=args.num_samplings,
        batch_size=args.batch_size, callbacks=callbacks, lmd=args.lmd
    )

    # test
    callbacks = []
    callbacks.append(MetricTracker({"fr": FoolingRatio()}))
    test_losses, test_metrics, macro_f1_score, micro_f1_score, rel_counter = attacker.test(
        g=args.g, net=args.net, predictor=args.predictor, target_net=args.target_net, exp_dir=args.exp_dir,
        embeds=args.embeds, xi=args.xi, search_scope=args.search_scope, num_samplings=args.num_samplings,
        callbacks=callbacks, auto_restore=args.auto_restore
    )
    for k, v in test_losses.items():
        args.logger.info(f"Test {k}: {v:.3f}")
    for k, v in test_metrics.items():
        args.logger.info(f"Test {k}: {v:.3f}")
    args.logger.info(f"Test Marco-F1 score: {macro_f1_score:.3f}")
    args.logger.info(f"Test Mirco-F1 score: {micro_f1_score:.3f}")
    for k, v in rel_counter.items():
        args.logger.info(f"Number of flipped edges in the meta relation {k}: {v}")