import glob
import os, os.path as osp
import pickle
from contextlib import nullcontext
from typing import Dict, List, Any
from logzero import logger

import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers.trainer_pt_utils import IterableDatasetShard

from argparse import Namespace
from torch.cuda.amp import autocast

from transformers import (
    AutoTokenizer,
    GPTJForCausalLM,
    AutoModelForCausalLM, 
    GPT2LMHeadModel,
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoConfig,
)
import json


def generate_from_dataset(
                generator, dataset, args, data_name=f'aug_train_texts.jsonl',
                config=None):
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    save_path = osp.join(save_dir, data_name)
    dataset.set_loop_data(data_name)

    num_samples = args.num_samples
    config.top_p = args.top_p
    config.temperature = args.temperature
    config.do_sample=True

    min_length = args.min_length
    max_length = args.max_length

    collator = dataset.collator
    if args.world_size > 1:
        logger.info(f"Sharding dataset for {args.world_size} processes")
        dataset = IterableDatasetShard(
            dataset,
            batch_size=args.per_device_eval_batch_size,
            drop_last=False,
            num_processes=args.world_size,
            process_index=args.process_index
        )
        save_path = save_path + f".{args.process_index}"
    total_iters = len(dataset) // args.per_device_eval_batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )
    # debug
    # for batch_ids, texts in tqdm(dataloader, disable=args.local_process_index > 0, total=total_iters):
    #     #print(f"rank {self.args.process_index} batch {example['text_id']}")
    #     logger.info(batch_ids)

    fout = open(save_path, 'w')
    cnt = 0
    for batch_ids, texts in tqdm(dataloader, disable=args.local_process_index > 0, total=total_iters):
        if cnt < 1:
            print("Example prompt:\n", texts[0])
        cnt += 1
        gen_text = generator.generate_batch(
                        texts, num_samples=num_samples,
                        min_length=min_length, max_length=max_length, 
                        gen_params=config, batch_size=args.gen_batch_size, 
                    )
        #generated.extend(gen_text)
        n = len(texts)
        for i in range(n):
            texts = []
            for j in range(num_samples):
                texts.append(gen_text[i*num_samples+j])
            out_dict = {"idx": f"{batch_ids[i]}", "text": texts}
            outs = json.dumps(out_dict)
            fout.write(f'{outs}\n')
        # for i, output_text in enumerate(gen_text):
        #     idx = i // args.num_samples
        #     text_id = batch_ids[idx]
        #     texts = []
        #     cnt = i % args.num_samples
        #     out_dict = {"idx": f"{text_id}_{cnt}", "text": output_text}
        #     outs = json.dumps(out_dict)
        #     fout.write(f'{outs}\n')
    fout.close()