import os, os.path as osp
import sys
import torch
import numpy as np
from genco.utils.data_utils import set_seed

from genco.arguments import ModelArguments, DataArguments, DenseTrainingArguments as TrainingArguments
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from logzero import logger
from collections import defaultdict

# data
from genco.selftrain.selftrain_dataset import load_and_proc_text, load_label_desc, load_text_data
from genco.selftrain.selftrain_cond_aug_dataset import load_and_proc_aug_text2
from genco.selftrain.selftrain_dataset import DataUtils

# model
from genco.selftrain.model_tools import load_model, load_ret_model
from genco.model.simcse import Model

# train
from xmclib.test.aug_selflearn_trainer import aug_gen_selftrain
from xmclib.test.selflearn_trainer import test_with_augment, test

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.output_dir = training_args.output_dir

    # load data
    data_dict = load_text_data(data_args)
    train_text, train_label, train_l2d = data_dict['train']
    test_text, test_label, test_l2d = data_dict['test']
    label_desc = data_dict['label_desc']

    # load augmented
    aug_dir = training_args.output_dir
    exp_name = data_args.exp_name
    aug_train_texts, aug_test_texts = load_and_proc_aug_text2(aug_dir, name=f"{exp_name}.alpaca-native.aug_{{}}_texts.jsonl") 

    device="cuda"   
    ret_model, ret_tokenizer = load_ret_model(model_args, device=device, dtype=torch.bfloat16)
    wrap_ret_model = Model(ret_model, ret_tokenizer, model_args)
    
    # dataloader
    data_utils = DataUtils(data_dict, data_args, training_args)
    logger.info(f"text len {model_args.max_text_length}")
    kwargs = {
        "aug_train_texts": aug_train_texts,
        "aug_test_texts": aug_test_texts,
        "generator": None,
        "gen_params": None,
        "gen_options": defaultdict(int),
    }
    set_seed(training_args.seed)
    dataloader = data_utils.get_conditional_generative_dataloader(**kwargs)

    # test
    pred, top_id, scores = dataloader.generate_pseudo_label(wrap_ret_model)
    #pred2, top_id2, scores2 = test_with_augment(wrap_ret_model, dataloader)
    res_dict = {
        "aug_pred": pred,
    }
    np.save(osp.join(aug_dir, "predict.npy"), res_dict)
    #test(model, data_utils, training_args)
    

if __name__ == '__main__':
    main()