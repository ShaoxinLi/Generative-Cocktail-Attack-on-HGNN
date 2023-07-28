#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import dgl
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from dgl.data import DGLDataset
from dgl.data.utils import makedirs, download, save_graphs, load_graphs, \
    generate_mask_tensor, idx2mask
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from sklearn.model_selection import train_test_split


def split_idx(samples, train_size, val_size, random_state=None):
    """将samples划分为训练集、测试集和验证集，需满足（用浮点数表示）：
    * 0 < train_size < 1
    * 0 < val_size < 1
    * train_size + val_size < 1
    :param samples: list/ndarray/tensor 样本集
    :param train_size: int or float 如果是整数则表示训练样本的绝对个数，否则表示训练样本占所有样本的比例
    :param val_size: int or float 如果是整数则表示验证样本的绝对个数，否则表示验证样本占所有样本的比例
    :param random_state: int, optional 随机数种子
    :return: (train, val, test) 类型与samples相同
    """
    train, val = train_test_split(samples, train_size=train_size, random_state=random_state)
    if isinstance(val_size, float):
        val_size *= len(samples) / len(val)
    val, test = train_test_split(val, train_size=val_size, random_state=random_state)
    return train, val, test


class DBLPFourAreaDataset(DGLDataset):
    """4领域DBLP学术网络数据集，只有一个异构图
    统计数据
    -----
    * 顶点：4057 author, 14328 paper, 20 conf, 7723 term
    * 边：19645 paper-author, 14328 paper-conf, 85810 paper-term
    * 类别数：4
    * author顶点划分：800 train, 400 valid, 2857 test
    属性
    -----
    * num_classes: 类别数
    * metapaths: 使用的元路径
    * predict_ntype: 预测顶点类型
    author顶点属性
    -----
    * feat: tensor(4057, 334)，关键词的词袋表示（来自HAN作者预处理的数据集）
    * label: tensor(4057)，0: DB, 1: DM, 2: AI, 3: IR
    * train_mask, val_mask, test_mask: tensor(4057)
    conf顶点属性
    -----
    * label: tensor(20)，类别为0~3
    """
    _url = "https://raw.githubusercontent.com/Jhy1993/HAN/master/data/DBLP_four_area/"
    _url2 = "https://pan.baidu.com/s/1Qr2e97MofXsBhUvQqgJqDg"
    _raw_files = [
        "readme.txt", "author_label.txt", "paper.txt", "conf_label.txt", "term.txt",
        "paper_author.txt", "paper_conf.txt", "paper_term.txt"
    ]
    _seed = 42

    def __init__(self):
        raw_dir = "/home/share/Datasets/openhgnn/dataset"
        super().__init__("DBLP_four_area", self._url, raw_dir=raw_dir)

    def download(self):
        if not os.path.exists(self.raw_path):
            makedirs(self.raw_path)
        for file in self._raw_files:
            download(self.url + file, os.path.join(self.raw_path, file))

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + "_dgl_graph.bin"), [self.g])

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + "_dgl_graph.bin"))
        self.g = graphs[0]
        for k in ("train_mask", "val_mask", "test_mask"):
            self.g.nodes["author"].data[k] = self.g.nodes["author"].data[k].bool()

    def process(self):
        self.authors, self.papers, self.confs, self.terms, \
            self.paper_author, self.paper_conf, self.paper_term = self._read_raw_data()
        self._filter_nodes_and_edges()
        self._lemmatize_terms()
        self._remove_stopwords()
        self._reset_index()

        self.g = self._build_graph()
        self._add_ndata()

    def _read_raw_data(self):
        authors = self._read_file("author_label.txt", names=["id", "label", "name"], index_col="id")
        papers = self._read_file("paper.txt", names=["id", "title"], index_col="id", encoding="cp1252")
        confs = self._read_file("conf_label.txt", names=["id", "label", "name", "dummy"], index_col="id")
        terms = self._read_file("term.txt", names=["id", "name"], index_col="id")
        paper_author = self._read_file("paper_author.txt", names=["paper_id", "author_id"])
        paper_conf = self._read_file("paper_conf.txt", names=["paper_id", "conf_id"])
        paper_term = self._read_file("paper_term.txt", names=["paper_id", "term_id"])
        return authors, papers, confs, terms, paper_author, paper_conf, paper_term

    def _read_file(self, filename, names, index_col=None, encoding="utf8"):
        return pd.read_csv(
            os.path.join(self.raw_path, filename), sep="\t", names=names, index_col=index_col,
            keep_default_na=False, encoding=encoding
        )

    def _filter_nodes_and_edges(self):
        """过滤掉不与学者关联的顶点和边"""
        self.paper_author = self.paper_author[self.paper_author["author_id"].isin(self.authors.index)]
        paper_ids = self.paper_author["paper_id"].drop_duplicates()
        self.papers = self.papers.loc[paper_ids]
        self.paper_conf = self.paper_conf[self.paper_conf["paper_id"].isin(paper_ids)]
        self.paper_term = self.paper_term[self.paper_term["paper_id"].isin(paper_ids)]
        self.terms = self.terms.loc[self.paper_term["term_id"].drop_duplicates()]

    def _lemmatize_terms(self):
        """对关键词进行词形还原并去重"""
        lemmatizer = WordNetLemmatizer()
        lemma_id_map, term_lemma_map = {}, {}
        for index, row in self.terms.iterrows():
            lemma = lemmatizer.lemmatize(row["name"])
            term_lemma_map[index] = lemma_id_map.setdefault(lemma, index)
        self.terms = pd.DataFrame(
            list(lemma_id_map.keys()), columns=["name"],
            index=pd.Index(lemma_id_map.values(), name="id")
        )
        self.paper_term.loc[:, "term_id"] = [
            term_lemma_map[row["term_id"]] for _, row in self.paper_term.iterrows()
        ]
        self.paper_term.drop_duplicates(inplace=True)

    def _remove_stopwords(self):
        """删除关键词中的停止词"""
        stop_words = sklearn_stopwords.union(nltk_stopwords.words("english"))
        self.terms = self.terms[~(self.terms["name"].isin(stop_words))]
        self.paper_term = self.paper_term[self.paper_term["term_id"].isin(self.terms.index)]

    def _reset_index(self):
        """将顶点id重置为0~n-1"""
        self.authors.reset_index(inplace=True)
        self.papers.reset_index(inplace=True)
        self.confs.reset_index(inplace=True)
        self.terms.reset_index(inplace=True)
        author_id_map = {row["id"]: index for index, row in self.authors.iterrows()}
        paper_id_map = {row["id"]: index for index, row in self.papers.iterrows()}
        conf_id_map = {row["id"]: index for index, row in self.confs.iterrows()}
        term_id_map = {row["id"]: index for index, row in self.terms.iterrows()}

        self.paper_author.loc[:, "author_id"] = [author_id_map[i] for i in self.paper_author["author_id"].to_list()]
        self.paper_conf.loc[:, "conf_id"] = [conf_id_map[i] for i in self.paper_conf["conf_id"].to_list()]
        self.paper_term.loc[:, "term_id"] = [term_id_map[i] for i in self.paper_term["term_id"].to_list()]
        for df in (self.paper_author, self.paper_conf, self.paper_term):
            df.loc[:, "paper_id"] = [paper_id_map[i] for i in df["paper_id"]]

    def _build_graph(self):
        pa_p, pa_a = self.paper_author["paper_id"].to_list(), self.paper_author["author_id"].to_list()
        pc_p, pc_c = self.paper_conf["paper_id"].to_list(), self.paper_conf["conf_id"].to_list()
        pt_p, pt_t = self.paper_term["paper_id"].to_list(), self.paper_term["term_id"].to_list()
        return dgl.heterograph({
            ("paper", "pa", "author"): (pa_p, pa_a),
            ("author", "ap", "paper"): (pa_a, pa_p),
            ("paper", "pc", "conf"): (pc_p, pc_c),
            ("conf", "cp", "paper"): (pc_c, pc_p),
            ("paper", "pt", "term"): (pt_p, pt_t),
            ("term", "tp", "paper"): (pt_t, pt_p)
        })

    def _add_ndata(self):
        _raw_file2 = os.path.join(self.raw_dir, "DBLP4057_GAT_with_idx.mat")
        if not os.path.exists(_raw_file2):
            raise FileNotFoundError("请手动下载文件 {} 提取码：6b3h 并保存到 {}".format(
                self._url2, _raw_file2
            ))
        mat = sio.loadmat(_raw_file2)
        self.g.nodes["author"].data["feat"] = torch.from_numpy(mat["features"]).float()
        self.g.nodes["author"].data["label"] = torch.tensor(self.authors["label"].to_list())

        n_authors = len(self.authors)
        train_idx, val_idx, test_idx = split_idx(np.arange(n_authors), 800, 400, self._seed)
        self.g.nodes["author"].data["train_mask"] = generate_mask_tensor(idx2mask(train_idx, n_authors))
        self.g.nodes["author"].data["val_mask"] = generate_mask_tensor(idx2mask(val_idx, n_authors))
        self.g.nodes["author"].data["test_mask"] = generate_mask_tensor(idx2mask(test_idx, n_authors))

        self.g.nodes["conf"].data["label"] = torch.tensor(self.confs["label"].to_list())

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.name + "_dgl_graph.bin"))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("This dataset has only one graph")
        return self.g

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return 4

    @property
    def metapaths(self):
        return [["ap", "pa"], ["ap", "pc", "cp", "pa"], ["ap", "pt", "tp", "pa"]]

    @property
    def predict_ntype(self):
        return "author"
    
    
def load_dblp(device):
    dataset = DBLPFourAreaDataset()
    g = dataset[0]
    g = g.int()
    g = g.to(device, non_blocking=True)

    g.category = "author"
    g.num_classes = len(torch.unique(g.nodes[g.category].data["label"]))
    g.train_idxs = torch.nonzero(g.nodes[g.category].data["train_mask"], as_tuple=False).squeeze().int()
    g.val_idxs = torch.nonzero(g.nodes[g.category].data["val_mask"], as_tuple=False).squeeze().int()
    g.test_idxs = torch.nonzero(g.nodes[g.category].data["test_mask"], as_tuple=False).squeeze().int()
    g.labels = g.nodes[g.category].data["label"].int()
    return g


class DBLPFourAreaDatasetPaper(DGLDataset):
    """4领域DBLP学术网络数据集，只有一个异构图
    统计数据
    -----
    * 顶点：4057 author, 14328 paper, 20 conf, 7723 term
    * 边：19645 paper-author, 14328 paper-conf, 85810 paper-term
    * 类别数：4
    * author顶点划分：800 train, 400 valid, 2857 test
    属性
    -----
    * num_classes: 类别数
    * metapaths: 使用的元路径
    * predict_ntype: 预测顶点类型
    author顶点属性
    -----
    * feat: tensor(4057, 334)，关键词的词袋表示（来自HAN作者预处理的数据集）
    * label: tensor(4057)，0: DB, 1: DM, 2: AI, 3: IR
    * train_mask, val_mask, test_mask: tensor(4057)
    conf顶点属性
    -----
    * label: tensor(20)，类别为0~3
    """
    _url = "https://raw.githubusercontent.com/Jhy1993/HAN/master/data/DBLP_four_area/"
    _url2 = "https://pan.baidu.com/s/1Qr2e97MofXsBhUvQqgJqDg"
    _raw_files = [
        "readme.txt", "author_label.txt", "paper.txt", "conf_label.txt", "term.txt",
        "paper_author.txt", "paper_conf.txt", "paper_term.txt"
    ]
    _seed = 42

    def __init__(self):
        raw_dir = "/home/share/Datasets/openhgnn/dataset"
        super().__init__("DBLP_four_area", self._url, raw_dir=raw_dir)

    def download(self):
        if not os.path.exists(self.raw_path):
            makedirs(self.raw_path)
        for file in self._raw_files:
            download(self.url + file, os.path.join(self.raw_path, file))

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + "_dgl_graph_paper.bin"), [self.g])

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + "_dgl_graph_paper.bin"))
        self.g = graphs[0]
        for k in ("train_mask", "val_mask", "test_mask"):
            self.g.nodes["author"].data[k] = self.g.nodes["author"].data[k].bool()
        for k in ("train_mask", "val_mask", "test_mask"):
            self.g.nodes["paper"].data[k] = self.g.nodes["paper"].data[k].bool()

    def process(self):
        self.authors, self.papers, self.confs, self.terms, \
        self.paper_author, self.paper_conf, self.paper_term = self._read_raw_data()
        self._filter_nodes_and_edges()
        self._lemmatize_terms()
        self._remove_stopwords()
        self._reset_index()

        self.g = self._build_graph()
        self._add_ndata()

    def _read_raw_data(self):
        authors = self._read_file("author_label.txt", names=["id", "label", "name"], index_col="id")
        papers = self._read_file("paper.txt", names=["id", "title"], index_col="id", encoding="cp1252")
        confs = self._read_file("conf_label.txt", names=["id", "label", "name", "dummy"], index_col="id")
        terms = self._read_file("term.txt", names=["id", "name"], index_col="id")
        paper_author = self._read_file("paper_author.txt", names=["paper_id", "author_id"])
        paper_conf = self._read_file("paper_conf.txt", names=["paper_id", "conf_id"])
        paper_term = self._read_file("paper_term.txt", names=["paper_id", "term_id"])
        return authors, papers, confs, terms, paper_author, paper_conf, paper_term

    def _read_file(self, filename, names, index_col=None, encoding="utf8"):
        return pd.read_csv(
            os.path.join(self.raw_path, filename), sep="\t", names=names, index_col=index_col,
            keep_default_na=False, encoding=encoding
        )

    def _filter_nodes_and_edges(self):
        """过滤掉不与学者关联的顶点和边"""
        self.paper_author = self.paper_author[self.paper_author["author_id"].isin(self.authors.index)]
        paper_ids = self.paper_author["paper_id"].drop_duplicates()
        self.papers = self.papers.loc[paper_ids]
        self.paper_conf = self.paper_conf[self.paper_conf["paper_id"].isin(paper_ids)]
        self.paper_term = self.paper_term[self.paper_term["paper_id"].isin(paper_ids)]
        self.terms = self.terms.loc[self.paper_term["term_id"].drop_duplicates()]

    def _lemmatize_terms(self):
        """对关键词进行词形还原并去重"""
        lemmatizer = WordNetLemmatizer()
        lemma_id_map, term_lemma_map = {}, {}
        for index, row in self.terms.iterrows():
            lemma = lemmatizer.lemmatize(row["name"])
            term_lemma_map[index] = lemma_id_map.setdefault(lemma, index)
        self.terms = pd.DataFrame(
            list(lemma_id_map.keys()), columns=["name"],
            index=pd.Index(lemma_id_map.values(), name="id")
        )
        self.paper_term.loc[:, "term_id"] = [
            term_lemma_map[row["term_id"]] for _, row in self.paper_term.iterrows()
        ]
        self.paper_term.drop_duplicates(inplace=True)

    def _remove_stopwords(self):
        """删除关键词中的停止词"""
        stop_words = sklearn_stopwords.union(nltk_stopwords.words("english"))
        self.terms = self.terms[~(self.terms["name"].isin(stop_words))]
        self.paper_term = self.paper_term[self.paper_term["term_id"].isin(self.terms.index)]

    def _reset_index(self):
        """将顶点id重置为0~n-1"""
        self.authors.reset_index(inplace=True)
        self.papers.reset_index(inplace=True)
        self.confs.reset_index(inplace=True)
        self.terms.reset_index(inplace=True)
        author_id_map = {row["id"]: index for index, row in self.authors.iterrows()}
        paper_id_map = {row["id"]: index for index, row in self.papers.iterrows()}
        conf_id_map = {row["id"]: index for index, row in self.confs.iterrows()}
        term_id_map = {row["id"]: index for index, row in self.terms.iterrows()}

        self.paper_author.loc[:, "author_id"] = [author_id_map[i] for i in self.paper_author["author_id"].to_list()]
        self.paper_conf.loc[:, "conf_id"] = [conf_id_map[i] for i in self.paper_conf["conf_id"].to_list()]
        self.paper_term.loc[:, "term_id"] = [term_id_map[i] for i in self.paper_term["term_id"].to_list()]
        for df in (self.paper_author, self.paper_conf, self.paper_term):
            df.loc[:, "paper_id"] = [paper_id_map[i] for i in df["paper_id"]]

        paper_label = pd.read_csv("/home/share/Datasets/openhgnn/dataset/DBLP_four_area/paper_label.txt", sep="\t", names=["title", "area"], encoding="utf8")
        titles, areas = paper_label.title.tolist(), paper_label.area.tolist()
        self.paper_ids = []
        for title in titles:
            idx = self.papers.title.tolist().index(title)
            self.paper_ids.append(paper_id_map[self.papers.id.tolist()[idx]])
        self.paper_labels = []
        for title, area in zip(titles, areas):
            if area == "Database":
                self.paper_labels.append(0)
            elif area == "Data Mining":
                self.paper_labels.append(1)
            elif area == "AI":
                self.paper_labels.append(2)
            elif area == "Information Retrieval":
                self.paper_labels.append(3)
            else:
                print(title, area)
                assert False

    def _build_graph(self):
        pa_p, pa_a = self.paper_author["paper_id"].to_list(), self.paper_author["author_id"].to_list()
        pc_p, pc_c = self.paper_conf["paper_id"].to_list(), self.paper_conf["conf_id"].to_list()
        pt_p, pt_t = self.paper_term["paper_id"].to_list(), self.paper_term["term_id"].to_list()
        return dgl.heterograph({
            ("paper", "pa", "author"): (pa_p, pa_a),
            ("author", "ap", "paper"): (pa_a, pa_p),
            ("paper", "pc", "conf"): (pc_p, pc_c),
            ("conf", "cp", "paper"): (pc_c, pc_p),
            ("paper", "pt", "term"): (pt_p, pt_t),
            ("term", "tp", "paper"): (pt_t, pt_p)
        })

    def _add_ndata(self):
        _raw_file2 = os.path.join(self.raw_dir, "DBLP4057_GAT_with_idx.mat")
        if not os.path.exists(_raw_file2):
            raise FileNotFoundError("请手动下载文件 {} 提取码：6b3h 并保存到 {}".format(
                self._url2, _raw_file2
            ))
        mat = sio.loadmat(_raw_file2)
        self.g.nodes["author"].data["feat"] = torch.from_numpy(mat["features"]).float()
        self.g.nodes["author"].data["label"] = torch.tensor(self.authors["label"].to_list())

        n_authors = len(self.authors)
        train_idx, val_idx, test_idx = split_idx(np.arange(n_authors), 800, 400, self._seed)
        self.g.nodes["author"].data["train_mask"] = generate_mask_tensor(idx2mask(train_idx, n_authors))
        self.g.nodes["author"].data["val_mask"] = generate_mask_tensor(idx2mask(val_idx, n_authors))
        self.g.nodes["author"].data["test_mask"] = generate_mask_tensor(idx2mask(test_idx, n_authors))

        self.g.nodes["conf"].data["label"] = torch.tensor(self.confs["label"].to_list())

        self.g.nodes["paper"].data["label"] = torch.tensor(self.paper_labels)
        n_papers = len(self.papers)
        train_idx, val_idx, test_idx = split_idx(self.paper_ids, 800, 400, self._seed)
        self.g.nodes["paper"].data["train_mask"] = generate_mask_tensor(idx2mask(train_idx, n_papers))
        self.g.nodes["paper"].data["val_mask"] = generate_mask_tensor(idx2mask(val_idx, n_papers))
        self.g.nodes["paper"].data["test_mask"] = generate_mask_tensor(idx2mask(test_idx, n_papers))

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.name + "_dgl_graph_paper.bin"))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("This dataset has only one graph")
        return self.g

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return 4

    @property
    def metapaths(self):
        return [["ap", "pa"], ["ap", "pc", "cp", "pa"], ["ap", "pt", "tp", "pa"]]

    @property
    def predict_ntype(self):
        return "paper"


def load_dblp_paper(device):
    dataset = DBLPFourAreaDatasetPaper()
    g = dataset[0]
    g = g.int()
    g = g.to(device, non_blocking=True)

    g.category = "paper"
    g.num_classes = len(torch.unique(g.nodes[g.category].data["label"]))
    g.train_idxs = torch.nonzero(g.nodes[g.category].data["train_mask"], as_tuple=False).squeeze().int()
    g.val_idxs = torch.nonzero(g.nodes[g.category].data["val_mask"], as_tuple=False).squeeze().int()
    g.test_idxs = torch.nonzero(g.nodes[g.category].data["test_mask"], as_tuple=False).squeeze().int()
    g.labels = g.nodes[g.category].data["label"].int()
    return g
