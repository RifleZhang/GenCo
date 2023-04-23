from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from transformers import default_data_collator

@dataclass
class SimpleCollator():
    def __call__(self, features):
        res = []
        for l in range(len(features[0])):
            res.append([f[l] for f in features])
        return res

def make_prompt(text, instruction):
    # shorten text, so that the total number of tokens < 256
    words = text.split()
    if len(words) > 100:
        words = words[:100]
        text = " ".join(words)

    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{text}

### Response:"""
    
class TextDataset(IterableDataset):

    def __init__(self, data_utils, instruction=None):
        super().__init__()
        self.data_utils = data_utils 
        self.args = data_utils.args
        self.collator=SimpleCollator()
        self.loop_data = self.data_utils.train_texts
        self.instruction = instruction
    
    def set_loop_data(self, data_name):
        if "train" in data_name:
            self.loop_data = self.data_utils.train_texts
        elif "test" in data_name:
            self.loop_data = self.data_utils.test_texts
    
    def num_label(self):
        return len(self.data_utils.label_desc)

    def __len__(self):
        return len(self.loop_data)
        
    def __iter__(self):
        for i, text in enumerate(self.loop_data):
            if self.instruction is not None:
                text = make_prompt(text, self.instruction)
            # words = text.split()
            # if len(words) > 100:
            #     words = words[:100]
            #     text = " ".join(words)
            yield i, text

class ConditionalTextDataset(TextDataset):

    def __init__(self, data_utils, pred=None, instruction="Discuss the <c> aspects of the article."):
        super().__init__(data_utils)
        self.instruction = instruction
        self.data_desc = self.data_utils.label_desc
        self.pred = pred # pseudo label by sentence encoder
        if self.pred is not None:
            assert len(self.pred) == len(self.loop_data)
        
    def get_prompt(self, x, label):
        #return x.strip() + " " + self.instruction.replace("<c>", label)
        return make_prompt(x.strip(), self.instruction.replace("<c>", label.lower()))

    def __len__(self):
        if self.pred is None:
            return len(self.loop_data) * len(self.data_utils.data_desc)
        else:
            return len(self.loop_data)

    def __iter__(self):
        if self.pred is None: # no pseudo label, generate for all
            for i, t in enumerate(self.loop_data):
                for j, ll in enumerate(self.data_desc):
                    yield f"{i}_{j}", self.get_prompt(t, ll)
        else: # generate for pseudo label
            for i, t in enumerate(self.loop_data):
                j = self.pred[i]
                ll = self.data_desc[j]
                yield f"{i}_{j}", self.get_prompt(t, ll)