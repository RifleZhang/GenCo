import os, sys, os.path as osp, time
import re
import numpy as np
from tqdm import tqdm
from logzero import logger
from collections import defaultdict
import scipy.sparse as smat
import pandas as pd
from typing import List, Dict
import json
import pickle
import random
import torch

# IO
def load_data(text_path, encoding='utf-8'):
    """
    load textual data from file
    """
    with open(text_path, encoding=encoding) as fp:
        texts = fp.readlines()
    return [t.strip() for t in texts]

def save_data(filename, data):
    """
    write textual data to file
    """
    with open(filename, 'w') as fout:
        for d in data:
            fout.write(str(d) + '\n')
    fout.close()

def save_jsonl(path: str, entries: List[Dict]):
    with open(path, 'w', encoding='utf8') as fh:
        for entry in entries:
            fh.write(f'{json.dumps(entry)}\n')

def load_jsonl(path: str) -> List[Dict]:
    pairs = []
    with open(path, 'r', encoding='utf8') as fh:
        for line in fh:
            pairs.append(json.loads(line))
    return pairs

def load_json(path):
    with open(path, 'r', encoding='utf8') as fh:
        content = json.load(fh)
    return content

def save_json(path, content):
    with open(path, 'w', encoding='utf8') as fh:
        json.dump(content, fh, indent=4)

# IO Embedding
def load_embedding(emb_path):
    with open(emb_path, 'rb') as f:
        emb, indices = pickle.load(f)
    return emb.astype("float32"), indices

def load_embedding_from_dir(emb_dir):
    path = osp.join(emb_dir, "embeddings.corpus")
    if osp.exists(path):
        return load_embedding(path)
    emb = []
    indices = []
    rank = 0
    while True:
        path = osp.join(emb_dir, f"embeddings.corpus.rank.{rank}")
        if not osp.exists(path):
            break
        cur_emb, cur_indices = load_embedding(path)
        emb.extend(cur_emb)
        indices.extend(cur_indices)
        rank += 1
    return np.array(emb), indices

def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# label processing
def label_list_from_all_labels(all_labels):
    """
    get label id list ranked by freq
    
    Parameters
    ----------
    all_labels: list[list] labels in train, [dev], test

    Returns
    ----------
    label_list: list of label id (sorted from hi2lo)
    freq: corresponding frequency
    """
    label_freq_dict = defaultdict(int) # map: label -> freq
    for labels in all_labels:
        for ll in labels:
            label_freq_dict[ll] += 1
    label_list = np.array(list(label_freq_dict.keys()))
    freq = np.array(list(label_freq_dict.values()))
    sorted_idx = np.argsort(freq)[::-1] # high -> low
    label_list = label_list[sorted_idx]
    freq = freq[sorted_idx]
    return label_list, freq

def parse_multilabel(file_name):
    """
    Parse the label file for multi-label dataset
    labels are separate by space in each line
    
    Parameters
    ----------
    file_name: str, path to label file

    Returns
    ----------
    labels: list[list] of labels      
    """
    labels = []
    file = open(file_name, 'r')
    lines = file.readlines()
    file.close()
    for line in lines:
        label = line.strip()
        labels.append(label.split(' '))
    return labels

def label_to_sparse_matrix(labels, n_label):
    row, col = [], []
    for i,label in enumerate(labels):
        for ll in label: # ins i, label ll
            row.append(i)
            col.append(ll)
    data = np.ones(len(row))
    mtx = smat.csr_matrix((data, (row, col)), shape=(len(labels), n_label), dtype=int)
    return mtx

def get_label_map(label_list):
    return {label : i for i, label in enumerate(label_list)}

def label_freq_from_matrix(label_mtx):
    # get label frequency (sorted) from highest to lowest
    # the order is the same in label_text.txt
    return np.sort(np.array(np.sum(label_mtx, 0)).reshape(-1))[::-1]

def binarize_label(labels, label_map):
    """
    binarize labels

    Parameters
    ----------
    labels: list[list] of labels
    label_map: dict, map label to id

    Returns
    ----------
    label_binary: list[list] of binarized labels
    """
    label_binary = []
    for label in labels:
        row = []
        for ll in label:
            row.append(label_map[ll])
        label_binary.append(row)
    return label_binary

def sparse_to_list(target):
    """
    sparse matrix to list of labels
    """
    n, m = target.shape
    d = []
    indptr, indices = target.indptr, target.indices
    for lo, hi in zip(indptr[:-1], indptr[1:]):
        d.append(indices[lo:hi])
    return d, m

def get_inv_propensity(train_y, a=0.55, b=1.5):
    """
    get inverse propensity for each label (same parameter as in AttentionXML)
    """
    n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)

def freq_table(label_freq):
    """
    label frequency table (aggregate label with same count), shape: max_count

    Parameters
    ----------
    label_freq: list of label frequency

    Returns
    ----------
    tbl: label frequency table
    tbl_idx: index information of original categories
    """
    label_freq = label_freq.astype(int)
    ll = max(label_freq)
    # print(f"max len {ll}")
    tbl = np.zeros(ll + 1)
    tbl_idx = defaultdict(list)
    for i, f in enumerate(label_freq):
        tbl[f] += 1
        tbl_idx[f].append(i)
    return tbl, tbl_idx