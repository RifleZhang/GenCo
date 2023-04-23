from argparse import ArgumentParser
import os, os.path as osp
import random
from tqdm import tqdm
import json
from datetime import datetime
import glob
import logzero
from logzero import logger
from timeit import default_timer
from collections import defaultdict
import pandas as pd
import string
import re

def normalize_text(orig_text):
    text = orig_text.strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S*@\S*\s?', '', text).replace("\\n", " ").replace("\n", " ")
    text = re.sub(' +', ' ', text)
    try:
        if text[-1] not in string.punctuation:
            text += "."
        if text[-1] == "\\":
            text = text[:-1] + "."
    except:
        return " ", True
    return text, False

def get_data(data_dir, data_prefix, save_dir, num_per_class):
    x_path = osp.join(data_dir, f"{data_prefix}.txt")
    y_path = osp.join(data_dir, f"{data_prefix}_labels.txt")
    with open(x_path, 'r') as fin:
        X = fin.readlines()
    
    bad_idx = []
    if not "amazon" in data_dir:
        for i in tqdm(range(len(X))):
            X[i], bad = normalize_text(X[i])
            if bad:
                bad_idx.append(i)
        X = [_.replace(" #39;", '\'') for _ in X]
    label_dict = defaultdict(list)
    with open(y_path, 'r') as fin:
        for idx, line in enumerate(fin):
            if idx in bad_idx:
                continue
            label = int(line.strip())
            label_dict[label].append(idx)
    return X, label_dict

def get_data_csv(data_dir, data_prefix, save_dir, num_per_class):
    data_path = osp.join(data_dir, f"{data_prefix}.csv")
    bad_idx = []
    if "ag_news" in data_dir:
        df = pd.read_csv(data_path)
        titles = df['Title'].tolist()
        texts = df['Description'].tolist()
        labels = df['Class Index'].tolist()
        X = [f"{t}. {d}" for t, d in zip(titles, texts)]
        labels = [int(l)-1 for l in labels]
    elif "yahoo" in data_dir:
        df = pd.read_csv(data_path, header=None)
        q1 = df[1].tolist()
        q2 = df[2].tolist()
        texts = df[3].tolist()
        X = []
        for i in range(len(q1)):
            s = ""
            if str(q1[i]) != "nan":
                s += str(q1[i]) + " "
            # if str(q2[i]) != "nan":
            #     s += str(q2[i]) + " "
            if str(texts[i]) != "nan":
                s += str(texts[i])
            X.append(s)
        labels = df[0].tolist()
        labels = [int(l)-1 for l in labels]
    
    for i in tqdm(range(len(X))):
        X[i], bad = normalize_text(X[i])
        if bad:
            bad_idx.append(i)
    label_dict = defaultdict(list)
    for idx, ll in enumerate(labels):
        if idx not in bad_idx:
            label_dict[ll].append(idx)
    return X, label_dict

def main(data_dir, data_prefix, save_dir, num_per_class, **kwargs):
    label_name_path = osp.join(data_dir, "crafted_label_names.txt")
    # count lines of label_names.txt
    with open(label_name_path, 'r') as fin:
        num_classes = len(fin.readlines())
    logger.info(f"num_classes: {num_classes}, sample per class: {num_per_class}")

    if "csv" in data_dir:
        X, label_dict = get_data_csv(data_dir, data_prefix, save_dir, num_per_class)
    else:
        X, label_dict = get_data(data_dir, data_prefix, save_dir, num_per_class)
    
    # set random seed
    random.seed(args.seed)
    for label, idx_list in label_dict.items():
        random.shuffle(idx_list)
        label_dict[label] = idx_list[:args.num_per_class]
    
    # save
    x_out = open(osp.join(args.save_dir, f'{data_prefix}.txt'), 'w')
    y_out = open(osp.join(args.save_dir, f'{data_prefix}_labels.txt'), 'w')
    os.makedirs(args.save_dir, exist_ok=True)
    for label, idx_list in label_dict.items():
        for idx in idx_list:
            x_out.write(X[idx].strip() + '\n')
            y_out.write(f'{label}' + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--data_prefix', default='train')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--num_per_class', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=100)

    args = parser.parse_args()
    main(**vars(args))