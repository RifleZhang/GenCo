import os, os.path as osp
import sys
import torch
import numpy as np

from genco.arguments import ModelArguments, DataArguments, DenseTrainingArguments as TrainingArguments
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from transformers import GenerationConfig

# data
from genco.utils.data_utils import set_seed
from genco.selftrain.selftrain_dataset import load_and_proc_text, load_label_desc, load_text_data
from genco.selftrain.selftrain_cond_aug_dataset import load_and_proc_aug_text2
from genco.selftrain.selftrain_dataset import DataUtils

# model
from genco.selftrain.model_tools import GeneratorTool, load_model, load_ret_model, load_gen_model
from genco.model.simcse import Model

# train
from genco.selftrain.aug_selftrainer import test_with_augment, test
from genco.selftrain.aug_selftrainer import cond_aug_gen_selftrain_filter

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.output_dir = training_args.output_dir

    device = model_args.device 
    gen_device = model_args.gen_device
    exp_name = data_args.exp_name

    schedule = np.array(training_args.schedule.split(",")).astype(float)
    sample_size = np.array(training_args.sample_size.split(",")).astype(float) 
    initial_sample = training_args.initial_sample
    max_iter = training_args.max_iter

    # load data
    data_dict = load_text_data(data_args)

    # load augmented
    aug_dir = training_args.output_dir
    aug_train_texts, aug_test_texts = load_and_proc_aug_text2(aug_dir, name=f"{exp_name}.alpaca-native.aug_{{}}_texts.jsonl")

    # set up generative model
    conf = GenerationConfig.from_pretrained(model_args.gen_model_name)
    conf.top_p = 0.95
    conf.temperature=0.8
    conf.do_sample=True
    print("generation configuration:\n", conf)

    # gen_tool=None
    int8=False # running int8 on single GPU is slower than fp16
    gen_model, gen_tokenizer = load_gen_model(model_args, large_model=int8, device=gen_device)
    gen_tool = GeneratorTool(gen_model, gen_tokenizer)

    # dataloader
    data_utils = DataUtils(data_dict, data_args, training_args)
    gen_tool_option = {
        "min_length": 64,
        "max_length": 128,
        "batch_size": 45,
        "num_samples": 20,
    }

    kwargs = {
        "aug_train_texts": aug_train_texts,
        "aug_test_texts": aug_test_texts,
        "generator": gen_tool,
        "gen_params": conf,
        "gen_options": gen_tool_option,
    }
    dataloader = data_utils.get_conditional_generative_dataloader(**kwargs)
    print(dataloader.p2l)

    # load cached conditional generated texts
    gen_dict_path = osp.join(aug_dir, f"{exp_name}.alpaca-native.cond_gen_dict.npy")
    try:
        gen_dict = np.load(gen_dict_path, allow_pickle=True).item()
    except:
        gen_dict = {}
    dataloader.gen_text_dict = gen_dict
    
    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    elif training_args.bf16:
        dtype = torch.bfloat16

    # init sentence encoder
    ret_model, ret_tokenizer = load_ret_model(model_args, device=device, dtype=dtype)
    wrap_ret_model = Model(ret_model, ret_tokenizer, model_args)
    # a lag model to refresh index
    inf_dtype = torch.float16
    lag_model, _ = load_ret_model(model_args, device=device, dtype=inf_dtype) 
    wrap_model_lag = Model(lag_model, ret_tokenizer, model_args)
    wrap_model_lag.dtype=inf_dtype
    for param in wrap_model_lag.parameters():
        param.requires_grad = False  # not update by gradient

    # parameters for self-training
    # schedule = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0]) 
    # sample_size = np.array([0.075, 0.1, 0.125, 0.15, 0.175, 0.25]) 
    ds_args = {
        "train_batch_size": 32,
        "eval_batch_size": 500,
        "max_iter": max_iter,
        "learning_rate": 1e-5,
        "p_num": 3,
        "gen_num_samples": 20,
        "num_instance_per_class": initial_sample,
        "schedule": schedule,
        "sample_size": sample_size,
    }
    # train
    set_seed(training_args.seed)
    wrap_ret_model.set_temperature(0.1)
    cond_aug_gen_selftrain_filter(wrap_ret_model, wrap_model_lag, dataloader, training_args, fun_name='cond_aug_filter', **ds_args)

    # save the conditionally generated texts
    np.save(gen_dict_path, dataloader.gen_text_dict)

    # test
    _ = test(wrap_ret_model, data_utils)
    _ = test_with_augment(wrap_ret_model, dataloader)

    model_path = osp.join(aug_dir, f"{exp_name}.alpaca-native.cond_gen_model.pt")
    torch.save(wrap_ret_model.state_dict(), model_path)
    #test(model, data_utils, training_args)
    

if __name__ == '__main__':
    main()