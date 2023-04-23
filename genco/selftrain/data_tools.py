from datasets import load_dataset, Dataset, disable_caching
from torch.utils.data import IterableDataset, get_worker_info
import torch
import numpy as np
from transformers import PreTrainedTokenizer
import os
import json
from argparse import Namespace

from genco.arguments import DataArguments
from genco.utils.data_utils import load_data
from genco.utils.normalize_text import normalize

LABEL_TAG="<c>"
TEXT_TAG="<x>"

def to_cuda(inputs, is_tensor=True, device="cuda"):
    if isinstance(inputs, list):
        if not is_tensor:
            inputs = [torch.tensor(e) for e in inputs]
        return [e.to(device) for e in inputs]
    elif isinstance(inputs, dict):
        for e in inputs:
            if not is_tensor:
                inputs[e] = torch.tensor(inputs[e])
            inputs[e] = inputs[e].to(device)
    else:
        if not is_tensor:
            inputs = torch.tensor(inputs)
        inputs = inputs.to(device)
    return inputs

def process_text(text):
    text = text.replace(' #39;', '\'')
    text = text.replace('#39;', '\'')
    text = text.replace('<br />', '').replace('\n\n', '')
    text = normalize(text)
    return text.strip()

def build_prompt_text(prompt, label_tag, text_tag, label, text, text_max_len=128):
    words = text.replace('<br />', ' ').split()[:text_max_len]
    text = " ".join(words)
    return prompt.replace(label_tag, label).replace(text_tag, text)

default_args = Namespace(text_max_len=256)

class PromptBuilder:
    def __init__(self, prompts, label_name, label_tag, text_tag, args=default_args):
        self.prompts = prompts
        self.label_name = label_name
        self.label_tag = label_tag
        self.text_tag = text_tag

        self.args = args
        self.text_max_len = args.text_max_len
    
    def build_prompt_text(self, prompt, label, text):
        text = process_text(text)
        return prompt.replace(self.label_tag, label).replace(self.text_tag, text)
    
    def build_prompt_for_texts_and_labels(self, prompt, texts, labels):
        prompt_texts = []
        for text in texts:
            for label in labels:
                prompt_text = self.build_prompt_text(prompt, label, text)
                prompt_texts.append(prompt_text)
        return prompt_texts
        
    def build_label_prompt(self):
        prompt_texts = []
        for label in self.label_name:
            for prompt in self.prompts:
                prompt_text = self.build_prompt_text(prompt, label, label)
                prompt_texts.append(prompt_text)
        return prompt_texts

    @classmethod
    def load(cls, prompt_path, label_path, label_tag=LABEL_TAG, text_tag=TEXT_TAG):
        fin_label = open(label_path, "r")
        label_name = [label.strip() for label in fin_label]
        fin_prompt = open(prompt_path, "r")
        prompts = [prompt.strip() for prompt in fin_prompt]
        
        return PromptBuilder(prompts, label_name, label_tag, text_tag)