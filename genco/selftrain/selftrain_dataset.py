import numpy as np
import re, string
import random
import torch
import math, copy, time
import os, os.path as osp
from tqdm import tqdm
import sys
from transformers import RobertaTokenizer,RobertaTokenizerFast
from collections import Counter
import json
import pickle
import scipy.sparse as smat
from collections import defaultdict
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd
from logzero import logger

from xmclib.utils.data_utils import load_data, save_data, load_json, save_json, load_jsonl, save_jsonl
from datasets import load_dataset, Dataset, disable_caching
from torch.utils.data import IterableDataset, get_worker_info

def load_text_data(data_args):
    text_dir = data_args.data_dir
    train_text, train_label, train_l2d = load_and_proc_text(text_dir, "train")
    test_text, test_label, test_l2d = load_and_proc_text(text_dir, "test")
    label_desc, num_label = load_label_desc(text_dir)
    # return a dict
    return {
        "train": (train_text, train_label, train_l2d),
        "test": (test_text, test_label, test_l2d),
        "label_desc": label_desc
    }

def complement_augment_text(text, generated, num_samples=5):
    generated = np.array(generated).reshape(len(text), num_samples)
    aug_text = []
    for i, x in enumerate(text):
        aug_text.append(x)
        for a in generated[i]:
            aug_text.append(a)
    return aug_text

def augment_text(text, generated, num_samples=5):
    generated = np.array(generated).reshape(len(text), num_samples)
    aug_text = []
    for i, x in enumerate(text):
        for a in generated[i]:
            aug_text.append(x + " " + a.strip())
    return aug_text

# def augment_text(text, generated, num_samples=5):
#     generated = np.array(generated).reshape(len(text), num_samples)
#     aug_text = []
#     for i, x in enumerate(text):
#         x = x.split()[:80]
#         x = " ".join(x)
#         for a in generated[i]:
#             aug_text.append(x + ". " + a.strip())
#     return aug_text

def augment_sent(sent, aug_sent, num_samples=5):
    compete_sent = []
    aug_sent = np.array(aug_sent, dtype='object').reshape(len(sent), num_samples)
    for i, x in enumerate(sent):
        for a in aug_sent[i]:
            compete_sent.append(x + a)
    return compete_sent
    
def load_and_proc_text(path, mode='train'):
    text_path = os.path.join(path, f'{mode}.txt')
    label_path = os.path.join(path, f'{mode}_labels.txt')
    texts = load_data(text_path)
    labels = load_data(label_path)
    labels = [int(l) for l in labels]
    
    l_count = defaultdict(int)
    for l in labels:
        l_count[l] += 1
    l2d = defaultdict(list)
    for i,label in enumerate(labels):
        l2d[label].append(i)
    
    avg_length = np.mean([len(t.strip().split()) for t in texts])
    print(f'{path} average length: {avg_length}')
    print(l_count)
    return texts, labels, l2d

def load_label_desc(data_dir):
    descs = [line.strip() for line in open(os.path.join(data_dir, 'crafted_label_names.txt'))]
    label_num = len(descs)
    return descs, label_num

def dest2prompt(desc, data_name):
    label_vocab = defaultdict(list)
    for l,line in enumerate(desc):
        if "dbpedia" in data_name:
            for w in line.strip().split('and'):
                label_vocab[l].append(w.strip())
        else:
            label_vocab[l].append(line.strip())
        # for w in line.strip().split('and'):
        #     label_vocab[l].append(w.strip())
    
    p2l,w2l = {},{}

    if 'yahoo' in data_name:
        for l in label_vocab:
            for w in label_vocab[l]:
                p2l['Category: {}.'.format(w)] = l
                p2l['It is about {}.'.format(w.lower())] = l
    elif 'ag' in data_name:
        for l in label_vocab:
            for w in label_vocab[l]:
                p2l['Category: {} news.'.format(w)] = l
                p2l['{} news.'.format(w)] = l
                #p2l['It is about {} news.'.format(w.lower())] = l
                #p2l['The topic is {}.'.format(w.lower())] = l
                #p2l['It is about {}.'.format(w.lower())] = l
    elif 'dbpedia' in data_name:
        for l in label_vocab:
            for w in label_vocab[l]:
                p2l['Category: {}.'.format(w)] = l
                p2l['It is about {}.'.format(w)] = l
    elif 'imdb' in data_name:
        for l in label_vocab:
            for w in label_vocab[l]:
                p2l['It was a {} movie.'.format(w)] = l
                p2l['In summary, the movie is {}.'.format(w)] = l
    elif 'yelp' in data_name:
        for l in label_vocab:
            for w in label_vocab[l]:
                if w == 'excellent':
                    p2l['It was an {} restaurant.'.format(w)] = l
                else:
                    p2l['It was a {} restaurant.'.format(w)] = l
                p2l['In summary, the restaurant is {}.'.format(w)] = l
    elif 'amazon' in data_name:
        for l in label_vocab:
            for w in label_vocab[l]:
                p2l['It is a {} product.'.format(w)] = l
                p2l['In summary, the product is {}.'.format(w)] = l

    l2p = defaultdict(list)
    for p in p2l:
        l2p[p2l[p]].append(p)
    return p2l, l2p

def get_list(d, ids):
    return [d[i] for i in ids]

def get_list_from_aug(d, ids, num_samples=5):
    ret = []
    for i in ids:
        for j in range(num_samples):
            ret.append(d[i*num_samples + j])
    return ret

def get_list_aug_sents(sents, aug_sents, ids, num_samples=5):
    ret = []
    for i in ids:
        ss = sents[i]
        j = random.randint(0, num_samples-1)
        aug = aug_sents[i*num_samples + j]
        ret.append(ss + aug)
    return ret

def get_list_all_aug_sents(sents, aug_sents, ids, num_samples=5):
    ret = []
    for i in ids:
        ss = sents[i]
        for j in range(num_samples):
            aug = aug_sents[i*num_samples + j]
            ret.append(ss + aug)
    return ret

def load_aug_text(aug_text_path):
    tmp = load_jsonl(aug_text_path)
    aug_text = []
    for x in tmp:
        aug_text.append(x['text'])
    return aug_text

def dict2list(d):
    return [d[i] for i in range(len(d))]

def unpack_field(ds, train_data, test_data, label_desc):
    ds.train_texts, ds.train_labels, ds.train_l2d = train_data
    ds.test_texts, ds.test_labels, ds.test_l2d = test_data
    ds.label_desc = label_desc
    ds.label_num = len(label_desc)
    ds.p2l, ds.l2p = dest2prompt(ds.label_desc, ds.data_name)

class DataUtils:
    def __init__(self, data_dict, args, training_args):
        super().__init__()
        self.data_dict = data_dict
        self.data_name = osp.basename(args.data_dir)
        self.args = args
        self.training_args = training_args

        self.train_data = data_dict['train'] # train_text, train_label, train_l2d = data_dict['train']
        self.test_data = data_dict['test'] # test_text, test_label, test_l2d = data_dict['test']
        #self.label = data_dict['label'] # label_desc, num_label = data_dict['label']
        self.label_desc = data_dict['label_desc']
        unpack_field(self, self.train_data, self.test_data, self.label_desc)
        if "train_sents" in data_dict:
            self.train_sents = data_dict['train_sents']
        self.output_dir=training_args.output_dir
        self.train_num = len(self.train_texts)
        
    def reset_label_desc(self, label_desc):
        self.label_desc = label_desc
        self.label_num = len(label_desc)
        self.p2l, self.l2p = dest2prompt(self.label_desc, self.data_name)

    # def get_sup_dataloader(self, **kwargs):
    #     from .supervise_dataset import SuperviseDataLoader
    #     return SuperviseDataLoader(
    #         data_dict=self.data_dict, args=self.args, training_args=self.training_args,
    #         **kwargs
    #     )

    # def get_selflearn_dataloader(self, **kwargs):
    #     from .selflearn_dataset import SelfLearnDataLoader
    #     return SelfLearnDataLoader(
    #         data_dict=self.data_dict, args=self.args, training_args=self.training_args,
    #         **kwargs
    #     )
    
    # def get_augselflearn_dataloader(self, **kwargs):
    #     from .selflearn_aug_dataset import AugSelfLearnDataLoader
    #     return AugSelfLearnDataLoader(
    #         data_dict=self.data_dict, args=self.args, training_args=self.training_args,
    #         **kwargs
    #     )
    
    # def get_generative_selflearn_dataloader(self, **kwargs):
    #     from .selflearn_aug_dataset import GenerativeSelfLearnDataLoader
    #     return GenerativeSelfLearnDataLoader(
    #         data_dict=self.data_dict, args=self.args, training_args=self.training_args,
    #         **kwargs
    #     )

    # def get_aug_pesco_dataloader(self, **kwargs):
    #     from .pesco_aug_dataset import AugPESCODataLoader
    #     return AugPESCODataLoader(
    #         data_dict=self.data_dict, args=self.args, training_args=self.training_args,
    #         **kwargs
    #     )
    
    def get_conditional_generative_dataloader(self, **kwargs):
        from .selftrain_cond_aug_dataset import CondontionalGenerativeSelfTrainDataLoader
        return CondontionalGenerativeSelfTrainDataLoader(
            data_dict=self.data_dict, args=self.args, training_args=self.training_args,
            **kwargs
        )
    
    def get_label_prompt(self):
        return self.p2l, self.l2p

