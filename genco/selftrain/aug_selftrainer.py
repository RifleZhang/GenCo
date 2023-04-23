import numpy as np
import sys, os, os.path as osp
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast
from transformers import get_linear_schedule_with_warmup
from logzero import logger
from tqdm import tqdm
from .data_tools import to_cuda
from genco.selftrain.selftrain_dataset import augment_text
from genco.selftrain.model_tools import get_context
from genco.selftrain.eval_tool import get_per_category_acc
EPS=1e-6

def test(model, data_utils, sample=False):
    print('testing')
    return model.generate_pseudo_label(data_utils.test_texts, data_utils.p2l, data_utils.test_labels)

def test_with_augment(model, data_utils):
    print('testing with augment')
    aug_test = augment_text(data_utils.test_texts, data_utils.aug_test_texts, 5)
    return model.generate_pseudo_label_for_augmented_text(aug_test, data_utils.p2l, data_utils.test_labels)

def momentum_update_lag_encoder(encoder, encoder_lag, m=0.99):
    for param, param_lag in zip(
        encoder.parameters(), encoder_lag.parameters()
    ):
        param_lag.data = param_lag.data * m + param.data * (1.0 - m)

def cond_aug_gen_selftrain_filter(model, model_lag, dataloader, training_args, **kwargs):
    dataloader.begin_epoch(**kwargs)
    model.train()
    tokenizer = model.tokenizer
    device = model.get_device()
    learning_rate = kwargs['learning_rate']
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=EPS)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, dataloader.max_iter)

    if 'fun_name' in kwargs:
        fun_name = kwargs['fun_name']
    else:
        fun_name = "get_train_iter"
    train_iter = getattr(dataloader, fun_name)
    max_length = model.max_text_length
    cnt = 0
    for querys, supports, target in train_iter(model, model_lag):
        q0 = tokenizer(querys[0], padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        q1 = tokenizer(querys[1], padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        s0 = tokenizer(supports[0], padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        s1 = tokenizer(supports[1], padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        # input_text = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
        # input_prompt = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt")
        q0, q1, s0, s1 = to_cuda(q0, device=device), to_cuda(q1, device=device), to_cuda(s0, device=device), to_cuda(s1, device=device)
        #input_text, input_prompt = to_cuda(input_text, device=device), to_cuda(input_prompt, device=device)
        target = to_cuda(target, device=device)
        #prompt_loss, scl_loss = model.supervised_loss_fn(input_text, input_prompt, target)
        context = get_context(training_args=training_args)
        with context:
            prompt_loss = model.selflearn_contrastive_loss(q0, s0, target[0])
            scl_loss = model.selflearn_contrastive_loss(q1, s1, target[1])
            loss = prompt_loss + scl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        #momentum_update_lag_encoder(model, model_lag, m=0.99)

        if cnt % 50 == 0:
            print(f"iter {cnt}, prompt loss {prompt_loss:.4f}, scl loss {scl_loss:.4f}")
        cnt += 1
