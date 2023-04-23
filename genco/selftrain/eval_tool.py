# general
import os, sys, time, os.path as osp
import math
import numpy as np
import glob
from collections import defaultdict 
import torch
import numpy as np
import random

import sys
import os, os.path as osp
import torch
import math
import pickle
from tqdm import tqdm

import json


def accuracy(pred, target):
    # overall accuracy
    pred = np.array(pred)
    target = np.array(target)
    return np.mean(pred == target)

def get_per_category_acc(pred, l2d=None, label=None, verbose=True):
    # per category accuracy
    label = np.array(label)
    if l2d is None:
        assert label is not None
        l2d = defaultdict(list)
        for i, ll in enumerate(label):
            l2d[ll].append(i)
    pred = np.array(pred)
    per_category_recall = np.zeros(len(l2d))
    per_category_prec = np.zeros(len(l2d))
    per_category_f1 = np.zeros(len(l2d))
    for k, v in l2d.items():
        v = np.array(v)
        recall = np.mean(k == pred[v])
        prec_idx = pred == k 
        prec = np.mean(k == label[prec_idx])
        f1 = 2 * prec * recall / (prec + recall)
        per_category_recall[k] = recall
        per_category_prec[k] = prec
        per_category_f1[k] = f1

    if verbose:
        acc = accuracy(pred, label)
        print(f"overall acc {acc*100:.2f}")
        for k, prec in enumerate(per_category_prec):
            recall = per_category_recall[k]
            f1 = per_category_f1[k]
            cnt = np.sum(pred == k)
            print(f"label {k}, precision {prec*100:.2f}, recall {recall*100:.2f}, f1 {f1*100:.2f}, cnt {cnt}")
    return acc, per_category_prec, per_category_recall, per_category_f1

def get_ranklist_acc(pred, scores, test_label, precision_list=[10, 20, 30, 50, 70, 100], verbose=True):
    """
        ranklist by confidence score
        get per category topk precision according to ranked list.
    """
    # topk acc
    pred = np.array(pred)
    test_label = np.array(test_label)
    pred2id = defaultdict(list)
    pred2score = defaultdict(list)
    ret = np.zeros((scores.shape[-1], len(precision_list)))
    for i, p in enumerate(pred):
        pred2id[p].append(i)
        pred2score[p].append(scores[i][p])
    for k, v in pred2score.items():
        ss = pred2score[k]
        ss_idx = np.argsort(ss)[::-1]
        vv = np.array(pred2id[k])
        vv = vv[ss_idx]
        pred2id[k] = vv
        ss = np.array(ss)[ss_idx]
        pred2score[k] = ss
        for i, r in enumerate(precision_list):
            acc = accuracy(pred[vv[:r]], test_label[vv[:r]])
            ret[k][i] = acc
    
    if verbose:
        for k, v in enumerate(ret):
            print(f"label {k}")
            print(f"top id {pred2id[k][:10]}")
            print(f"top score ", end="")
            for j, s in enumerate(pred2score[k][:10]):
                print(f"{s*100:.1f}", end=" ")
            print()
            for j, acc in enumerate(v):
                print(f"{precision_list[j]} : {acc}", end="   ")
            print()
    return ret

def get_sample_acc(sample_l2d, test_label, verbose=True):
    """
        sample_l2d: per category sample,  label: list(doc id)
        test_label: ground truth, doc id : label
    """
    test_label = np.array(test_label)
    ret = {}
    if verbose:
        print(f"num sample per category: {len(sample_l2d[0])}")
    for k in range(len(sample_l2d)):
        v = sample_l2d[k]
        v = np.array(v)
        cnt = len(v)
        acc = np.mean(k == test_label[v])
        ret[k] = acc
    if verbose:
        for k in ret:
            print(f"label {k}, p@{cnt} {ret[k]}")
    return ret

def predict(a, b, num_label=4):
    score = a @ b.T
    ll = len(a)
    pred = score.reshape(ll, num_label, -1).mean(-1)
    return pred
def evaluate(pred, target):
    pp = np.argmax(pred, -1)
    print(pp)
    return np.sum(pp == target) / len(target) 

# error analysis
def sample_data(texts, labels, shot, l2d=None):
    texts = np.array(texts)
    labels = np.array(labels)
    if l2d is None:
        l2d = defaultdict(list)
        for i,label in enumerate(labels):
            l2d[label].append(i)
    sample_l2d = defaultdict(list)
    sample_texts, sample_labels = [], []
    for label in range(len(l2d)):
        indices = np.array(l2d[label])
        assert len(indices) > shot
        sample_idx = np.random.choice(indices, shot, replace=False)
        sample_l2d[label] = sample_idx
        sample_texts.extend(texts[sample_idx])
        sample_labels.extend(labels[sample_idx])
    return sample_texts, sample_labels, sample_l2d

def get_wrong_sample(pred, labels):
    """
        key: predicted as label
        value: idx
    """
    pred = np.array(pred)
    labels = np.array(labels)
    idx = pred != labels
    
    l2d = defaultdict(list)
    for i, val in enumerate(idx):
        if val:
            l2d[pred[i]].append(i)
    return l2d

def get_wrong_data(texts, labels, l2d, num_sample=5):
    texts = np.array(texts)
    labels = np.array(labels)
    
    sample_l2d = defaultdict(list)
    sample_texts, sample_labels = [], []
    for label in range(len(l2d)):
        indices = np.array(l2d[label])
        assert len(indices) >= num_sample
        sample_idx = np.random.choice(indices, num_sample, replace=False)
        sample_l2d[label] = sample_idx
        sample_texts.extend(texts[sample_idx])
        sample_labels.extend(labels[sample_idx])
    return sample_texts, sample_labels, sample_l2d

def get_wrong_data2(texts, labels, l2d, scores):
    texts = np.array(texts)
    labels = np.array(labels)
    
    sample_l2d = defaultdict(list)
    sample_texts, sample_labels = [], []
    for label in range(len(l2d)):
        indices = np.array(l2d[label])
        sample_idx = np.random.choice(indices, 1, replace=False)

        #assert len(indices) >= num_sample
        # ss = scores[indices]
        # tmp = np.sort(ss, -1)
        # delta = tmp[:, -1] - tmp[:, -2]
        # sorted_idx = np.argsort(delta)
        # sample_idx = indices[sorted_idx[9:10]]

        sample_l2d[label] = sample_idx
        sample_texts.extend(texts[sample_idx])
        sample_labels.extend(labels[sample_idx])
    return sample_texts, sample_labels, sample_l2d

def get_wrong_data_for_label(label, w_l2d, scores): 
    indices = np.array(w_l2d[label])
    #sample_idx = np.random.choice(indices, 1, replace=False)

    ss = scores[indices]
    tmp = np.sort(ss, -1)
    delta = tmp[:, -1] - tmp[:, -2]
    sorted_idx = np.argsort(delta)
    sample_idx = indices[sorted_idx]

    return sample_idx
