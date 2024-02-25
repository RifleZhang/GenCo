import numpy as np
import re, string
import random
import torch
import math, copy, time
import os, os.path as osp
from tqdm import tqdm
import sys
from collections import Counter
import json
import pickle
from collections import defaultdict
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd
from logzero import logger

from genco.utils.data_utils import load_data, save_data, load_json, save_json, load_jsonl, save_jsonl
from datasets import load_dataset, Dataset, disable_caching
from genco.selftrain.text_dataset import make_prompt
from .selftrain_dataset import DataUtils, get_list, get_list_from_aug, augment_text, complement_augment_text
from .data_tools import to_cuda
from .eval_tool import get_sample_acc, get_per_category_acc


def load_and_proc_aug_text2(aug_dir, name="aug_{}_texts.json"):
    aug_train_path = osp.join(aug_dir, name.format("train"))
    aug_test_path = osp.join(aug_dir, name.format("test"))
    
    aug_train = load_json(aug_train_path)
    aug_test = load_json(aug_test_path)

    return aug_train, aug_test

def load_and_proc_aug_text(aug_dir, load_sent=False):
    aug_train_path = osp.join(aug_dir, "aug_train_texts.json")
    aug_test_path = osp.join(aug_dir, "aug_test_texts.json")
    aug_train_sents_path = osp.join(aug_dir, "aug_train_sents.json")
    
    aug_train = load_json(aug_train_path)
    aug_test = load_json(aug_test_path)
    if load_sent:
        if not osp.exists(aug_train_sents_path):
            logger.info(f"process aug sents")
            aug_train_sents = sentence_seg(aug_train)
            save_json(aug_train_sents_path, aug_train_sents)
        else:
            aug_train_sents = load_json(aug_train_sents_path)
    else:
        aug_train_sents = None
    return aug_train, aug_test, aug_train_sents

class CondontionalGenerativeSelfTrainDataLoader(DataUtils):
    def __init__(self, 
                data_dict, args, training_args,
                **kwargs
        ):
        super().__init__(data_dict, args, training_args)
    
        self.batch_size = self.training_args.per_device_train_batch_size
        self.train_num = len(self.train_texts)
        self.teacher_update = 0

        self.label_tag = "<c>"
        self.prompt = f"Discuss the {self.label_tag} aspects of the article."
        self.generator = kwargs['generator']
        self.gen_params = kwargs['gen_params']
        self.gen_options = kwargs['gen_options']
        self.gen_num_samples = self.gen_options['num_samples']
        self.gen_text_dict = {}

        if "aug_train_texts" in kwargs: # pass in aug data
            self.aug_train_texts = kwargs["aug_train_texts"]
            self.aug_test_texts = kwargs["aug_test_texts"]
        else: # load aug data
            aug_train, aug_test = load_and_proc_aug_text2(self.args.aug_dir)
            self.aug_train_texts = aug_train
            self.aug_test_texts = aug_test
        self.aug_train_num = len(self.aug_train_texts)
        self.num_samples = self.aug_train_num // self.train_num
        logger.info(f"augmented texts num: {self.aug_train_num}, num_samples: {self.num_samples}")

    def begin_epoch(self, train_batch_size, num_instance_per_class=10, 
                    max_iter=500, p_num=2, use_augment=True, **kwargs):
        self.batch_size = train_batch_size
        self.ins_per_c = num_instance_per_class
        self.max_iter = max_iter
        self.p_num = p_num
        self.use_augment = use_augment
        self.temperature = 0.1
        self.test_result = {}
        if "gen_num_samples" in kwargs:
            self.gen_num_samples = kwargs['gen_num_samples']
            self.gen_options['num_samples'] = kwargs['gen_num_samples']
        if "schedule" in kwargs:
            self.schedule = kwargs['schedule']
            self.sample_size = kwargs['sample_size']
        else:
            self.schedule = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0]) 
            self.sample_size = np.array([0.075, 0.1, 0.125, 0.15, 0.175, 0.25]) 
    
    def generate_pseudo_label(self, model, eval_batch_size=500, refresh_index=True):
        model.eval()
        train_complete_text = augment_text(self.train_texts, self.aug_train_texts, self.num_samples)
        #train_complete_text = complement_augment_text(self.train_texts, self.aug_train_texts, self.num_samples)
        pseudo_labels, Q, label_scores = model.generate_pseudo_label_for_augmented_text(
            train_complete_text, 
            self.p2l, labels=self.train_labels,
            batch_size=eval_batch_size, num_samples=self.num_samples)#, update_labels
        if refresh_index:
            self.pseudo_labels = pseudo_labels
            self.Q = torch.from_numpy(Q)
            self.label_scores = label_scores
        model.train()
        return pseudo_labels, Q, label_scores
    
    def generate_pseudo_label_baseline(self, model, eval_batch_size=500, refresh_index=True):
        model.eval()
        pseudo_labels,Q,label_scores = model.generate_pseudo_label(
            self.train_texts, 
            self.p2l, labels=self.train_labels,
            batch_size=eval_batch_size)
        if refresh_index:
            self.pseudo_labels = pseudo_labels
            self.Q = Q
            self.label_scores = label_scores
        model.train()
        return pseudo_labels, Q, label_scores

    def sample_topk(self, topk, verbose=True):
        if verbose:
            logger.info(f"sample top {topk}")
        sample_l2d = defaultdict(list)

        pred = np.array(self.pseudo_labels)
        scores = np.array(self.label_scores)
        pred2id = defaultdict(list)
        pred2score = defaultdict(list)
        
        for i, p in enumerate(pred):
            pred2id[p].append(i)
            pred2score[p].append(scores[i][p])
        for k, v in pred2score.items():
            ss = pred2score[k]
            ss_idx = np.argsort(ss)[::-1][:topk]
            # if verbose:
            #     last = ss_idx[-1]
            #     logger.info(f"score threshold for {k} is {ss[last]}")
            vv = np.array(pred2id[k])
            vv = vv[ss_idx] # select topk indices
            sample_l2d[k] = vv.tolist()
        return sample_l2d
    
    def resample(self, model, topk):
        self.generate_pseudo_label(model)
        model.train()
        return self.sample_topk(topk)

    def get_and_filter_cond_gen_texts(self, model, texts, pseudo_label, return_num=1, **kwargs):
        n, m = len(texts) // self.gen_num_samples, self.gen_num_samples
        # score-based selection
        aug_emb = model.encode(texts, batch_size=500)
        label_emb = model.encode_label_prompt(self.p2l)
        aug_emb, label_emb = aug_emb.float(), label_emb.float()
        aug_scores = aug_emb @ label_emb.T
        aug_scores = aug_scores.view(n, m, -1) # n text, m gen sample, num_labels
        
        probs = torch.softmax(aug_scores, dim=-1)
        aug_probs = probs[np.arange(n), :, pseudo_label]
        top_idx = torch.argsort(aug_probs, dim=-1, descending=True)[:, :5]
        top_idx = top_idx.cpu().numpy()
        texts = np.array(texts).reshape(n, m)
        selected = []
        Q = []
        for i in range(n):
            select_idx = np.random.choice(top_idx[i], return_num)
            selected.append(texts[i, select_idx])
            Q.append(aug_scores[i, select_idx])
            # selected.append(np.random.choice(texts[i, top_idx[i]], return_num))
        selected = np.array(selected).reshape(n, return_num)
        Q = torch.softmax(torch.cat(Q) / self.temperature, dim=-1)
        return selected, Q

    def get_cond_gen_texts(self, texts, return_num=1):
        n, m = len(texts) // self.gen_num_samples, self.gen_num_samples
        selected = []
        for xx in np.array(texts).reshape(n, m):
            selected.append(np.random.choice(xx, return_num))
        selected = np.array(selected).reshape(n, -1)
        return selected

    def augment_text_by_label(self, texts, labels):
        """
        text: list of text
        label: list of label
        """
        if len(texts) == 0:
            return []
        text_prompts = []
        labels = list(labels)
        for text, label in zip(texts, labels):
            #text_prompts.append(text + " " + self.prompt.replace(self.label_tag, self.label_desc[label]))
            instruction = self.prompt.replace(self.label_tag, self.label_desc[label])
            text_prompts.append(make_prompt(text, instruction))
        aug_texts = self.generator.generate_batch(text_prompts, gen_params=self.gen_params, **self.gen_options)
        return aug_texts

    # search mark: here
    def cond_aug_filter(self, model, model_lag):
        """
        move text2text contrastive and text2prompt contrastive to the same batch
        """
        # aug as query and another aug as support, plus label prompt as support
        logger.info(f"iteration: {self.max_iter}")
        logger.info(f"positive num: {self.p_num}")
        logger.info(f"initial num positive instance: {self.ins_per_c}")

        model_lag.eval()

        self.generate_pseudo_label(model_lag)
        sample_train_l2d = self.sample_topk(self.ins_per_c)
        get_per_category_acc(self.pseudo_labels, l2d=self.train_l2d, label=self.train_labels)
        get_sample_acc(sample_train_l2d, self.train_labels)

        p_num = self.p_num
        text_num = self.label_num * p_num
        select_sample = 1
        target = [torch.zeros(text_num * select_sample, self.label_num),torch.zeros(text_num, text_num)]
        for i in range(text_num * select_sample):
            l = i// (p_num * select_sample)
            target[0][i,l] = 1.0
        for i in range(text_num):
            l = i//p_num
            target[1][i,l*p_num:l*p_num+p_num] = 1.0


        schedule = np.array(self.schedule) * self.max_iter
        num_per_class = self.train_num // self.label_num
        sample_size = np.array(self.sample_size) * num_per_class
        schedule = schedule.astype(int)
        sample_size = sample_size.astype(int)
        logger.info(f"sample size: {sample_size}")
        logger.info(f"schedule: {schedule}")
        
        ss = 0
        for i in range(self.max_iter):
            orig_texts, pred_labels, gen_keys = [], [], []
            batch_keys = []
            batch_texts, batch_aug_texts = [], []
            for l in range(self.label_num):
                #np.random.seed(123)
                s_ids = np.random.choice(sample_train_l2d[l], p_num, replace=False)
                for s_id in s_ids:
                    batch_texts.append(self.train_texts[s_id])
                    # general augmentation
                    aug_train_texts = get_list_from_aug(self.aug_train_texts, [s_id], self.num_samples)
                    # batch_text.extend(aug_train_texts)
                    aug_train_texts = np.array(aug_train_texts)
                    rand_idx = np.random.choice(range(5), 1, replace=False)
                    batch_aug_texts.extend(aug_train_texts[rand_idx])

                # take keys to be generated in batch
                for s_id in s_ids:
                    kk = f"{s_id}_{l}"
                    if kk not in self.gen_text_dict:    
                        orig_texts.append(self.train_texts[s_id])
                        pred_labels.append(l)
                        gen_keys.append(kk)
                    batch_keys.append(kk)

            # batch generation
            if len(gen_keys) > 0:
                cond_aug_texts = self.augment_text_by_label(orig_texts, pred_labels)
                cond_aug_texts = np.array(cond_aug_texts).reshape(-1, self.gen_num_samples)
                for kk, aug_texts in zip(gen_keys, cond_aug_texts):
                    self.gen_text_dict[kk] = aug_texts
            # select cond gen texts
            #x_texts = []
            gen_texts = []
            pseudo_labels = []
            for kk in batch_keys:
                aug_texts = self.gen_text_dict[kk]
                gen_texts.extend(list(aug_texts))
                sid = int(kk.split("_")[0])
                #x_texts.append(self.train_texts[sid])
                pseudo_labels.append(int(kk.split("_")[1]))
            selected, Q = self.get_and_filter_cond_gen_texts(model_lag, gen_texts, pseudo_labels, return_num=select_sample)
            #import pdb; pdb.set_trace()
            selected = list(selected.reshape(-1))
            label_prompt = [np.random.choice(self.l2p[l], 1, replace=False)[0] for l in range(self.label_num)]
            
            yield [selected, batch_texts], [label_prompt, batch_aug_texts], [Q, target[1]]
            batch_texts, label_prompt, batch_aug_texts = [], [], []
            batch_keys = []

            if (i+1) >= schedule[ss]:
                # refresh index
                model_lag.load_state_dict(model.state_dict())
                model_lag = model_lag.to(dtype=model_lag.dtype)
                logger.info(f"iteration {i+1} refresh index")
                self.generate_pseudo_label(model_lag)
                sample_train_l2d = self.sample_topk(sample_size[ss])
                acc, *_ = get_per_category_acc(self.pseudo_labels, l2d=self.train_l2d, label=self.train_labels)
                self.test_result[i] = acc
                print()
                get_sample_acc(sample_train_l2d, self.train_labels)
                ss += 1