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
from genco.selftrain.selftrain_dataset import DataUtils
from genco.selftrain.text_dataset import TextDataset
from transformers import GenerationConfig

# model
from genco.selftrain.model_tools import load_gen_model, GeneratorTool
# generate
from genco.selftrain.generation import generate_from_dataset

def load_and_save_texts(save_dir, num_proc, data_name='aug_train_texts.jsonl'):
    save_path = osp.join(save_dir, data_name)
    all_texts = {}
    for i in range(num_proc):
        aug_path = osp.join(save_dir, f'{data_name}.{i}')
        texts = load_jsonl(aug_path)
        for tt in texts:
            k, v = tt['idx'], tt['text']
            all_texts[k] = v
    all_texts_sorted = sorted(all_texts.items(), key=lambda x: int(x[0]))
    texts_array = []
    for k, v in all_texts_sorted:
        texts_array.extend(v)
    #all_texts = [v for k, v in all_texts_sorted]
    save_json(save_path, texts_array)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.output_dir = training_args.output_dir

    # load data
    data_dict = load_text_data(data_args)

    # dataset
    data_utils = DataUtils(data_dict, data_args, training_args)
    #import pdb; pdb.set_trace()
    #if "dbpedia" in data_args.data_dir or "yahoo" in data_args.data_dir:
    dataset = TextDataset(data_utils, instruction="Elaborate the text in a few sentences.")
    #else:
    #   dataset = TextDataset(data_utils)

    local_id = training_args.local_rank
    # model
    # gen_tool=None
    device = "cuda"
    int8=False # running int8 on single GPU is slower than fp16
    if int8 and training_args.local_rank != -1:
        device = f"cuda:{local_id}"
    config = GenerationConfig.from_pretrained(model_args.gen_model_name)
    gen_model, gen_tokenizer = load_gen_model(model_args, large_model=int8, device=device)
    gen_tool = GeneratorTool(gen_model, gen_tokenizer)

    # generate
    set_seed(training_args.seed)
    model_name_short = model_args.gen_model_name.split('/')[-1]
    exp_name = data_args.exp_name
    generate_from_dataset(generator=gen_tool, dataset=dataset, args=training_args, 
                        data_name=f'{exp_name}.{model_name_short}.aug_train_texts.jsonl',
                        config=config)
    generate_from_dataset(generator=gen_tool, dataset=dataset, args=training_args, 
                        data_name=f'{exp_name}.{model_name_short}.aug_test_texts.jsonl',
                        config=config)
    #aug = gen_tool.generate_batch(cond_s_texts, num_samples=ns, min_length=32, max_length=96, gen_params=gen_params, batch_size=45)
  
    if training_args.world_size > 1:
        torch.distributed.barrier()
    #load_and_save_texts(training_args.output_dir, training_args.world_size, data_name='train')
    if training_args.local_rank == 0:
        load_and_save_texts(training_args.output_dir, training_args.world_size, data_name=f'{exp_name}.{model_name_short}.aug_train_texts.jsonl')
        load_and_save_texts(training_args.output_dir, training_args.world_size, data_name=f'{exp_name}.{model_name_short}.aug_test_texts.jsonl')
    torch.distributed.barrier()
    
if __name__ == '__main__':
    main()
    import time
    time.sleep(20)