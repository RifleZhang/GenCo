import glob
import os, os.path as osp
import pickle
from contextlib import nullcontext
from typing import Dict, List, Any
import logging
from argparse import Namespace

import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
#from transformers.trainer_pt_utils import IterableDatasetShard
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from logzero import logger


from transformers import (
    AutoTokenizer,
    AutoModel,
    #GPTJForCausalLM,
    AutoModelForCausalLM, 
    GPT2LMHeadModel,
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoConfig,
)
import json

default_args = Namespace(max_len=256)
default_gen_params = {}

def load_and_cache_model(model_name, cache_dir, margs={}):
    save_name = osp.basename(model_name)
    model_cache_dir = osp.join(cache_dir, save_name)
    model_path = osp.join(model_cache_dir, "model.pt")
    if osp.exists(model_path):
        logger.info(f"load model from {model_path}")
        model = torch.load(model_path)
        return model
    MODEL_CLASS= AutoModelForCausalLM
    # if model_name == "EleutherAI/gpt-j-6B":
    #     save_name = "gpt-j-6B"
    #     if margs == {}:
    #         margs = {"revision": "float16", "torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    #         # margs = {"revision": "float16", "torch_dtype": torch.float16}
    # else:
    if margs == {}:
        margs = {"torch_dtype": torch.float16}
    
    margs['cache_dir'] = model_cache_dir
    logger.info(f"load and cache model")
    model = MODEL_CLASS.from_pretrained(
        model_name, **margs
    )   
    torch.save(model, model_path)
    return model

def get_context(torch_dtype=None, training_args=None):
    if torch_dtype is None: # infer dtype from training_args
        assert training_args is not None
        if training_args.fp16:
            torch_dtype = torch.float16
        elif training_args.bf16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
    context = nullcontext()
    if torch_dtype == torch.float16:
        context = amp.autocast()
    elif torch_dtype == torch.bfloat16:
        context = amp.autocast(dtype=torch.bfloat16)
    return context

def load_ret_model(model_args, device="cuda", dtype=torch.float32):
    logger.info(f"ret model dtype: {dtype}")
    cache_dir = model_args.cache_dir
    ret_model_name = model_args.model_name_or_path
    ret_model_name_short = os.path.basename(ret_model_name)
    ret_model_cache_dir = osp.join(cache_dir, ret_model_name_short)
    ret_model = AutoModel.from_pretrained(ret_model_name, cache_dir=ret_model_cache_dir)
    ret_model = ret_model.to(device=device, dtype=dtype)
    ret_tokenizer = AutoTokenizer.from_pretrained(ret_model_name, cache_dir=ret_model_cache_dir)
    return ret_model, ret_tokenizer

def load_and_cache_large_model(model_name, cache_dir, device):
    MODEL_CLASS= AutoModelForCausalLM
    save_name = osp.basename(model_name)
    model_cache_dir = osp.join(cache_dir, save_name)
    #config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    # with init_empty_weights(config):
    #     model = MODEL_CLASS.from_config(config)
    
    #max_memory =  {1: "49GIB"}
    #device_map = infer_auto_device_map(model, max_memory=max_memory)
    # model = load_checkpoint_and_dispatch(
    #     model,
    #     model_cache_dir,
    #     device_map=device_map,
    # )
    # save_name = osp.basename(model_name)
    device_map = {'': int(device[-1])}
    print(device_map)
    margs = {"load_in_8bit": True, "device_map": device_map}
    model_cache_dir = osp.join(cache_dir, save_name)
    margs['cache_dir'] = model_cache_dir
    model = MODEL_CLASS.from_pretrained(
        model_name, **margs
    )   
    return model


def load_gen_model(model_args, large_model=False, device="cuda", margs={}):
    cache_dir = model_args.cache_dir
    gen_model_name = model_args.gen_model_name
    
    from transformers import LlamaTokenizer 
    gen_tokenizer = LlamaTokenizer.from_pretrained(gen_model_name, cache_dir=cache_dir)

    if large_model:
        gen_model = load_and_cache_large_model(gen_model_name, cache_dir=cache_dir, device=device)
    else:
        #margs = {"revision": "float16", "torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        gen_model = load_and_cache_model(gen_model_name, cache_dir=cache_dir, margs=margs)
        gen_model = gen_model.to(device)
    # if "llama" in gen_model_name:
    #     from transformers import LlamaTokenizer 
    #     gen_tokenizer = LlamaTokenizer.from_pretrained(gen_model_name, cache_dir=cache_dir)
    # else:
    #     gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name, cache_dir=cache_dir)
    return gen_model, gen_tokenizer

def load_model(model_args, large_model=False, device="cuda", gen_device="cuda", dtype=torch.float32):
    ret_model, ret_tokenizer = load_ret_model(model_args, device=device, dtype=dtype)
    gen_model, gen_tokenizer = load_gen_model(model_args, large_model=large_model, device=gen_device)
    return {
        "ret_model": (ret_model, ret_tokenizer),
        "gen_model": (gen_model, gen_tokenizer)
    }



class RetreiverTool:
    def __init__(self, model, tokenizer, args=default_args):
        #self.model = model.to(args.device)
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.max_len = args.max_len
        self.pooling = args.pooling # default simcse is pooler
    
    def embed(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if self.pooling == "average":
            last_hidden = model_output["last_hidden_state"]
            last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling == "cls":
            emb = model_output["last_hidden_state"][:, 0]
        elif self.pooling == "pooler":
            emb = model_output["pooler_output"]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    @torch.no_grad()
    def encode(self, texts):
        self.model.eval()
        feature = self.tokenizer(texts, 
            padding='max_length', truncation=True, max_length=self.max_len, 
            return_tensors="pt"
        ).to(self.model.device)
        embeddings = self.embed(**feature)
        return embeddings.cpu().detach().numpy()

    def encode_batch(self, texts, batch_size=512):
        self.model.eval()
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            embeddings.append(self.encode(batch_texts))
        return np.concatenate(embeddings, axis=0)

    def predict(self, emb_text, emb_label):
        pred = emb_text @ emb_label.T
        return pred
    
    @classmethod
    def load_model(cls, model_name, args, device="cuda:0"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        return cls(model, tokenizer, args)


class GeneratorTool:
    def __init__(self, model, tokenizer, args=default_args):
        #self.model = model.to(args.device)
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

        # set pad token
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == "":
            if self.tokenizer.eos_token is None or self.tokenizer.eos_token == "":
                self.tokenizer.eos_token = '</s>'
                #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                #self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.args = args
        self.max_len = args.max_len
        #self.model.config.pad_token_id = tokenizer.eos_token_
       
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    
    def enc(self, texts):
        texts = list(texts) # can't take np array
        return self.tokenizer(texts, 
            padding='max_length', truncation=True, max_length=self.max_len, 
            return_tensors="pt"
        ).to(self.model.device)
    
    def enc_for_generation(self, texts):
        texts = list(texts) # can't take np array
        inputs = self.tokenizer(texts, 
            padding='max_length', truncation=True, max_length=self.max_len, 
            return_tensors="pt"
        ).to(self.model.device)
        batch_size, seq_len = inputs['input_ids'].shape

        inputs['attention_mask'] = torch.flip(inputs['attention_mask'], dims=[1])
        shifts = seq_len - inputs['attention_mask'].sum(dim=-1)
        for batch_idx in range(batch_size):
            inputs['input_ids'][batch_idx] = inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())
        return inputs, seq_len

    @torch.no_grad()
    def calc_loss(self, model, texts):
        self.model.eval()
        batch = self.enc(texts)
        outputs = model(**batch)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = batch["input_ids"][..., 1:].contiguous()
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
            shift_labels.size())
        # pad position loss is 0
        avg_loss = loss.sum(-1) / (loss > 0).sum(-1)
        return avg_loss.tolist()

    @torch.no_grad()
    def generate(self, prompt_texts, 
                num_samples=1, min_length=16, max_length=64, 
                gen_params=default_gen_params):
        torch_dtype = self.model.config.torch_dtype
        context = get_context(torch_dtype)
        with context:
            batch, seq_len = self.enc_for_generation(prompt_texts)
            
            min_length = min_length + seq_len
            max_length = min(self.model.config.max_position_embeddings, max_length + seq_len)

            output_ids = self.model.generate(
                **batch,
                num_return_sequences=num_samples, min_length=min_length, max_length=max_length,
                generation_config=gen_params
            )
            # get generated text
            output_ids = output_ids[:, seq_len:]
            gen_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # for i, output_text in enumerate(gen_text):
        #     output_text = post_process(output_text, self.min_length)
        return gen_text

    def generate_batch(self, prompt_texts, 
                num_samples=1, min_length=16, max_length=64, 
                gen_params=default_gen_params, batch_size=32, verbose=False):
        # actual batch is 32 on gpu
        bsz = batch_size // num_samples
        gen_texts = []
        if verbose:
            loop = tqdm(range(0, len(prompt_texts), bsz))
        else:
            loop = range(0, len(prompt_texts), bsz)
        for i in loop:
            gen_texts.extend(self.generate(prompt_texts[i:i+bsz], num_samples, min_length, max_length, gen_params))
        return gen_texts

   