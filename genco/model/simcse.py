import numpy as np
import torch
import math
import torch.nn as nn
import math, copy, time
import os, os.path as osp
import torch.nn.functional as F
from torch_scatter import scatter
from transformers import AutoModel,AutoTokenizer
from logzero import logger
from genco.selftrain.data_tools import to_cuda
from genco.selftrain.model_tools import load_model, load_ret_model

class Similarity(nn.Module):
    def __init__(self, norm=True):
        super().__init__()
        self.norm = norm

    def forward(self, x, y, temp=1.0):
        x, y = x.float(), y.float() # for bfloat16 optimization, need to convert to float
        if self.norm:
            x = x / torch.norm(x, dim=-1, keepdim=True)
            y = y / torch.norm(y, dim=-1, keepdim=True)
        return torch.matmul(x, y.t()) / temp

class Model(nn.Module):
    @classmethod
    def load(cls, model_args):
        cache_dir=os.path.join(model_args.cache_dir, model_args.model_name_short)
        transformer = AutoModel.from_pretrained(model_args.model_name_or_path, cache_dir = cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir = cache_dir)
        return cls(transformer, tokenizer, model_args)

    def __init__(self, model, tokenizer, model_args=None):
        super().__init__()
        self.transformer = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        if model_args is not None:
            self.model_name = model_args.model_name_or_path
            model_name_short = os.path.basename(self.model_name)
            self.cache_dir = os.path.join(model_args.cache_dir, model_name_short)
        self.sim = Similarity(norm=True)
        self.pooling = model_args.pooling # default simcse is pooler
        self.temperature = 0.1
        self.max_text_length = model_args.max_text_length
    
    def set_temperature(self, temperature):
        self.temperature =temperature

    def get_device(self):
        return self.transformer.device

    def reload(self):
        device = self.get_device()
        self.transformer = AutoModel.from_pretrained(self.model_name, cache_dir = self.cache_dir).to(device)

    def forward(self, inputs):
        model_output = self.transformer(**inputs, output_hidden_states=True, return_dict=True)
        if self.pooling == "average":
            last_hidden = model_output["last_hidden_state"]
            attention_mask = inputs["attention_mask"]
            last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling == "cls":
            emb = model_output["last_hidden_state"][:, 0]
        elif self.pooling == "pooler":
            emb = model_output["pooler_output"]
        return emb

    def contrastive_loss(self, embed1, embed2, target, mask=None, clip_prob=None):
        logits = self.sim(embed1, embed2)
        if mask is not None:
            logits = logits.masked_fill(mask, float('-inf'))
        log_prob = F.log_softmax(logits, dim=-1)
        if mask is not None:
            return -log_prob[target.bool()].mean()
        mean_log_prob_pos = (target * log_prob).sum(1) / (target.sum(1)+1e-10)
        return -mean_log_prob_pos.mean()

    # numerical stability issue
    # use old need to change the mask logic
    # def contrastive_loss(self, embed1, embed2, target, mask=None, clip_prob=None):
    #     logits = self.sim(embed1, embed2)
    #     if mask is not None:
    #         exp_logits = mask*torch.exp(logits)
    #     else:
    #         exp_logits = torch.exp(logits)
    #     log_prob = logits - torch.log(exp_logits.sum(1,keepdim=True)+1e-10)
    #     if clip_prob is not None:
    #         log_prob = torch.clamp(log_prob,max=math.log(clip_prob))
    #     mean_log_prob_pos = (target * log_prob).sum(1) / (target.sum(1)+1e-10)
    #     return -mean_log_prob_pos.mean()

    def supervised_loss_fn(self, input_text, input_prompt, target):
        text_emb = self(input_text)
        prompt_emb = self(input_prompt)
        p_loss = self.contrastive_loss(text_emb, prompt_emb, target[0])
        #p_loss = self.contrastive_entropy_loss(text_emb, prompt_emb, target[0])
        batch_size = text_emb.size(0)
        # mask = torch.ones([batch_size, batch_size])
        # for i in range(mask.size(0)):
        #     mask[i,i] = 0.0
        mask = torch.eye(batch_size).bool()
        scl_loss = self.contrastive_loss(text_emb, text_emb, target[1], mask.to(self.get_device()))
        #scl_loss = self.contrastive_entropy_loss(text_emb, text_emb, target[1], mask.to(self.get_device()))
        return p_loss,scl_loss

    def selflearn_contrastive_loss(self, query, support, target):
        query_embs = self(query)
        key_embs = self(support)
        # mask = torch.ones([query_embs.shape[0], key_embs.shape[0]])
        # for i in range(mask.size(0)):
        #     mask[i,i] = 0.0
        loss = self.contrastive_loss(query_embs, key_embs, target)#, mask.to(self.get_device()))
        #loss = self.contrastive_entropy_loss(query_embs, key_embs, target)
        return loss
    
    def selflearn_contrastive_label_loss(self, query, p2l, target):
        device = self.get_device()
        id2l = torch.tensor([p2l[desc] for desc in p2l]).to(device)
        id2desc = [desc for desc in p2l]
        desc_embeds = self.encode(id2desc)
        label_emb = scatter(desc_embeds, id2l, dim=0, reduce='mean')

        text_emb = self(query)
        loss = self.contrastive_loss(text_emb, label_emb, target)
        return loss

    def batch_contrastive_loss(self, texts, target):
        text_emb = self(texts)
        batch_size = text_emb.size(0)

        # mask = torch.ones([batch_size, batch_size])
        # for i in range(mask.size(0)):
        #     mask[i,i] = 0.0

        mask = torch.eye(batch_size).bool()
        loss = self.contrastive_loss(text_emb, text_emb, target, mask.to(self.get_device()))
        return loss

    def zero_shot_loss_fn(self, batch, train_loss2=False):
        query_embs = self(batch['text'])
        key_embs = self(batch['pos_text'])
        #prompt_embs = self(batch['prompt'])
        loss1 = self.contrastive_loss(query_embs, key_embs, batch['t2p'])#, batch['mask_t2p'])
        #loss2 = self.contrastive_loss(query_embs, prompt_embs, batch['t2l'])
        #loss3 = self.contrastive_loss(query_embs, query_embs, batch['t2t'], batch['mask_t2t'])
        #loss4 = self.contrastive_loss(key_embs, key_embs, batch['t2t'], batch['mask_t2t'])
        #loss5 = self.contrastive_loss(key_embs, query_embs, batch['t2p'].transpose(0,1))
        loss = loss1 #+ loss5#+ 0.01*loss2
        return loss

    def generate_pseudo_label_for_augmented_text(self, input_texts, p2l, labels=None, num_samples=5, batch_size=500):
        self.eval()
        # id2l = [p2l[desc] for desc in p2l]
        device = self.get_device()
        id2l = torch.tensor([p2l[desc] for desc in p2l]).to(device)
        l_num = max([e+1 for e in id2l])
        # lp_count = torch.zeros(l_num)
        l_pred_count = np.zeros(l_num)
        # for desc in p2l:
        #     lp_count[p2l[desc]] += 1
        # id2desc = [desc for desc in p2l]
        # desc_embeds = self.encode(id2desc)
        desc_embeds = self.encode_label_prompt(p2l).float()

        real_len = len(input_texts) // num_samples
        batch_size = batch_size // num_samples * num_samples
        acc,result = [],[]
        P, Q = [], []
        probs = np.zeros(len(input_texts))
        d_id = 0
        for c_id in range(math.ceil(len(input_texts)/batch_size)):
            texts = input_texts[c_id*batch_size:(c_id+1)*batch_size]
            text_embeds = self.encode(texts)
            text_embeds = text_embeds.view(-1, num_samples, text_embeds.size(-1)).mean(1)
            # scores = self.sim(text_embeds, desc_embeds)
            #text_embeds, desc_embeds = text_embeds.float(), desc_embeds.float() # for bf16 inference
            text_embeds = text_embeds.float()
            scores = torch.matmul(text_embeds, desc_embeds.transpose(0,1))
            #scores = scores.view(-1, len(id2desc))

            #l_scores = scatter(scores, id2l, dim=-1, reduce='mean')
            q_scores = F.softmax(scores / self.temperature, dim=-1)
            #q_scores = scatter(q_scores, id2l, dim=-1, reduce='mean')
            Q.append(q_scores.detach().cpu().numpy())
            p_scores = F.softmax(scores, dim=-1)
            #p_scores = scatter(p_scores, id2l, dim=-1, reduce='mean')
            P.append(p_scores.detach().cpu().numpy())
            
            # calculate P, Q
            # q_scores = F.softmax(scores / self.temperature, dim=-1)
            # q_scores = scatter(q_scores, id2l, dim=-1, reduce='mean')
            # Q.append(q_scores.detach().cpu().numpy())
            # p_scores = F.softmax(scores, dim=-1)
            # p_scores = scatter(p_scores, id2l, dim=-1, reduce='mean')
            # P.append(p_scores.detach().cpu().numpy())

            preds = torch.argmax(p_scores, dim=-1).cpu().numpy()
            idx, cnts = np.unique(preds, return_counts = True)
            for i in range(len(idx)):
                l_pred_count[idx[i]] += cnts[i]
            result.extend(preds)
        P = np.concatenate(P, 0)
        Q = np.concatenate(Q, 0)
        if labels is not None:
            acc = np.mean(np.array(result)==np.array(labels))
            logger.info(l_pred_count)
            logger.info(f'accu:{acc*100:.2f}, cnt:{len(labels)}')
        return result, Q, P

    def generate_pseudo_label(self, input_texts, p2l, labels=None, batch_size=500):
        self.eval()
        device = self.get_device()
        id2l = torch.tensor([p2l[desc] for desc in p2l]).to(device)
        l_num = max([e+1 for e in id2l])
        lp_count = torch.zeros(l_num)
        l_pred_count = np.zeros(l_num)
        for desc in p2l:
            lp_count[p2l[desc]] += 1
        id2desc = [desc for desc in p2l]
        desc_embeds = self.encode(id2desc)

        acc,result = [],[]
        all_scores = []
        probs = np.zeros(len(input_texts))
        d_id = 0
        for c_id in range(math.ceil(len(input_texts)/batch_size)):
            texts = input_texts[c_id*batch_size:(c_id+1)*batch_size]
            text_embeds = self.encode(texts)
            #scores = self.sim(text_embeds, desc_embeds)
            scores = torch.matmul(text_embeds, desc_embeds.transpose(0,1))
            scores = scores.view(-1, len(id2desc))

            # l_scores = F.softmax(scores, dim=-1)
            # l_scores = scatter(l_scores, id2l, dim=-1, reduce='mean')
            
            l_scores = scatter(scores, id2l, dim=-1, reduce='mean')
            l_scores = F.softmax(l_scores, dim=-1)

            all_scores.append(l_scores.detach().cpu().float().numpy())
            preds = torch.argmax(l_scores, dim=-1).cpu().numpy()
            idx, cnts = np.unique(preds, return_counts = True)
            for i in range(len(idx)):
                l_pred_count[idx[i]] += cnts[i]
            result.extend(preds)
            probs = l_scores[range(len(preds)), preds].cpu().float().numpy()
        all_scores = np.concatenate(all_scores, 0)
        top_ids = np.argsort(probs)[::-1]
        if labels is not None:
            acc = np.mean(np.array(result)==np.array(labels))

        if labels is not None:
            logger.info(l_pred_count)
            logger.info(f'accu:{acc*100:.2f}, cnt:{len(labels)}')
        return result, top_ids, np.array(all_scores)

    def update_prompt(self, inputs):
        self.prompt_embs = []
        for i in range(len(inputs)):
            emb = self.transformer(**inputs[i], output_hidden_states=True, return_dict=True).pooler_output
            self.prompt_embs.append(emb)
    
    @torch.no_grad()
    def encode_label_prompt(self, p2l):
        self.eval()
        device = self.get_device()
        id2l = torch.tensor([p2l[desc] for desc in p2l]).to(device)
        id2desc = [desc for desc in p2l]
        desc_embeds = self.encode(id2desc)
        label_emb = scatter(desc_embeds, id2l, dim=0, reduce='mean')
        self.train()
        return label_emb

    @torch.no_grad()
    def encode(self, texts, batch_size=None, cpu=False):
        self.eval()
        embeddings = []
        text_ids = []
        def enc(texts):
            inputs = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=self.max_text_length)
            inputs = to_cuda(inputs, device=self.transformer.device)
            outputs = self.transformer(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            return outputs
        if batch_size is None or len(texts) <= batch_size:
            embeddings = enc(texts)
        else:
            for i in range(len(texts)//batch_size + 1):
                s = i*batch_size
                e = (i+1)*batch_size
                if s >= len(texts):
                    break
                outputs = enc(texts[s:e])
                embeddings.append(outputs)
            embeddings = torch.cat(embeddings, 0)
        if cpu:
            embeddings = embeddings.cpu()
        self.train()
        return embeddings

    # def contrastive_entropy_loss(self, embed1, embed2, target, mask=None, clip_prob=None):
    #     temperature = self.temperature
    #     logits = self.sim(embed1, embed2)
    #     soft_target = target / temperature
    #     print(soft_target)
    #     # if temperature > 0:
    #     #     target_logits = logits / temperature
    #     #     soft_target = torch.softmax(target_logits, dim=-1)
    #     # else:
    #     #     soft_target = target # fall back to hard target
    #     #soft_target = soft_target.detach()
    #     #print(soft_target)
        
    #     exp_logits = torch.exp(logits)
    #     if mask is not None:
    #         exp_logits = mask*torch.exp(logits)
    #     else:
    #         exp_logits = torch.exp(logits)
    #     log_prob = logits - torch.log(exp_logits.sum(1,keepdim=True)+1e-10)
    #     if clip_prob is not None:
    #         log_prob = torch.clamp(log_prob,max=math.log(clip_prob))
    #     mean_log_prob_pos = (soft_target * log_prob).sum(1) / (target.sum(1)+1e-10)
    #     return -mean_log_prob_pos.mean()