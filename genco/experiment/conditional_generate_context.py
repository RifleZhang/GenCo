# general
import os, sys, time, os.path as osp
import math
import numpy as np
import glob
from collections import defaultdict 
from tqdm import tqdm
from timeit import default_timer as timer
import json

from transformers import (
    AutoTokenizer,
    HfArgumentParser
)

import torch
import logzero
from logzero import logger
import argparse
from genco.arguments import ModelArguments, DataArguments, DenseTrainingArguments as TrainingArguments
# from xmclib.dataset import GenerationDataset
from genco.utils.data_utils import save_jsonl, set_seed, load_jsonl, save_json

# data
from genco.selftrain.selftrain_dataset import load_and_proc_text, load_label_desc, load_text_data
from genco.selftrain.selftrain_cond_aug_dataset import load_and_proc_aug_text2
from genco.selftrain.selftrain_dataset import DataUtils
from genco.selftrain.text_dataset import ConditionalTextDataset
from transformers import GenerationConfig

# model
from genco.selftrain.model_tools import load_gen_model, GeneratorTool
# generate
from genco.selftrain.generation import generate_from_dataset

def load_and_save_texts(aug_pred, save_dir, num_proc, data_name):
    save_path = osp.join(save_dir, f"{data_name}.cond_gen_dict.npy")
    all_texts = {}
    for i in range(num_proc):
        aug_path = osp.join(save_dir, f'{data_name}.cond_aug_train_texts.jsonl.{i}')
        texts = load_jsonl(aug_path)
        for tt in texts:
            k, v = tt['idx'], tt['text']
            all_texts[k] = v
    np.save(save_path, all_texts)
    # all_texts_sorted = sorted(all_texts.items(), key=lambda x: (int(x[0].split('_')[0]), int(x[0].split('_')[1])))
    # all_texts = [v for k, v in all_texts_sorted]
    # save_json(save_path, all_texts)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.output_dir = training_args.output_dir

    # load data
    data_dict = load_text_data(data_args)
    train_text, train_label, train_l2d = data_dict['train']
    test_text, test_label, test_l2d = data_dict['test']
    label_desc = data_dict['label_desc']
    aug_dir = training_args.output_dir
    precition = np.load(osp.join(aug_dir, "predict.npy"), allow_pickle=True).item()
    aug_pred = precition['aug_pred']

    # aug_train_texts, aug_test_texts = load_and_proc_aug_text2(aug_dir, name="t0.8l64.alpaca-native.aug_{}_texts.jsonl")
    # aug_train_texts, aug_test_texts, aug_train_sents = load_and_proc_aug_text(aug_dir)
    # sents = proc_and_load_sents(training_args.output_dir)
    # aug_sents = proc_and_load_sents(training_args.output_dir, aug=True, 
    #                                 texts=aug_train_texts)
    # data_dict['train_sents'] = sents
    # data_dict['train_aug_sents'] = aug_sents

    # dataset
    data_utils = DataUtils(data_dict, data_args, training_args)
    dataset = ConditionalTextDataset(data_utils, aug_pred)
    #import pdb; pdb.set_trace()

    # model
    # gen_tool=None # debug

    device = "cuda"
    config = GenerationConfig.from_pretrained(model_args.gen_model_name)
    gen_model, gen_tokenizer = load_gen_model(model_args, device=device)
    gen_tool = GeneratorTool(gen_model, gen_tokenizer)

    # generate
    set_seed(training_args.seed)
    exp_name = data_args.exp_name
    model_name_short = model_args.gen_model_name.split('/')[-1]
    generate_from_dataset(generator=gen_tool, dataset=dataset, args=training_args, 
                        data_name=f'{exp_name}.{model_name_short}.cond_aug_train_texts.jsonl',
                        config=config)

    if training_args.world_size <= 1:
        return
    
    if training_args.world_size > 1:
        torch.distributed.barrier()
    if training_args.local_rank == 0:
        load_and_save_texts(aug_pred, training_args.output_dir, training_args.world_size, 
                            data_name=f'{exp_name}.{model_name_short}')
    torch.distributed.barrier()
    
if __name__ == '__main__':
    main()
    import time
    time.sleep(20)