import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name_short: str = field(default=None)
    gen_model_name: str = field(default=None)
    gen_model_name_short: str = field(default=None)
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

    decay_rate: Optional[float]=field(
        default=0.6, metadata={"help": "Decay learning rate"}
    )
    pooling: Optional[str]=field(
        default="pooler", metadata={"help": "Pooling method"}
    )
    device: Optional[str]=field(
        default="cuda:1", metadata={"help": "Device to use"}
    )
    gen_device: Optional[str]=field(
        default="cuda:0", metadata={"help": "Device to use"}
    )
    max_text_length: int = field(default=128)

@dataclass
class DataArguments:
    # data dir, path, name related
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to single train file"}
    )
    eval_path: str = field(
        default=None, metadata={"help": "Path to eval file"}
    )
    query_path: str = field(
        default=None, metadata={"help": "Path to query file"}
    )
    corpus_path: str = field(
        default=None, metadata={"help": "Path to corpus file"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "Path to data directory"}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the single data file"}
    )
    processed_data_path: str = field(
        default=None, metadata={"help": "Path to processed data directory"}
    )
    context_path: str= field(default=None)
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    prompt_path: str = field(default=None)
    label_name_path: str = field(default=None)
    label_path: str = field(default=None)
    save_path: str = field(default=None)

    # text specific
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    q_max_len: int = field(
        default=128,
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    query_template: str = field(
        default="<text>",
        metadata={"help": "template for query"}
    )
    query_column_names: str = field(
        default="id,text",
        metadata={"help": "column names for the tsv data format"}
    )
    doc_template: str = field(
        default="Title: <title> Text: <text>",
        metadata={"help": "template for doc"}
    )
    doc_column_names: str = field(
        default="id,title,text",
        metadata={"help": "column names for the tsv data format"}
    )
    num_per_class: int = field(
        default=100000,
    )
    limit: int = field(
        default=None,
    )
    num_class: int = field(
        default=2,
    )
    exp_name: str = field(default="exp")
        

@dataclass
class DenseTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
    
    # retrieval 
    use_gpu: bool = field(default=False, metadata={"help": "Use GPU for encoding"})
    faiss_index_type: str = field(default="IndexFlatIP", metadata={"help" : "which type of faiss index to use, please see documentation here https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index. Options are [IndexHNSWFlat, IndexHNSWFlat] for now"})
    faiss_index_search_batch_size: int = field(default=1, metadata={"help" : "batch size to search faiss index with, seems on A100 machines if an exact match index is on gpu, batch_size > 1 is troublesome..."})
    faiss_topk: int = field(default=100, metadata={"help" : "how many results to return from faiss index"})

    load_large_model: bool = field(default=False)
    num_samples: int = field(default=5)
    encoding_mode: str = field(default=None)

    # generate
    generation_mode: str = field(default=None)

    load_large_model: bool = field(default=False)

    num_samples: int = field(default=5)

    top_k: int = field(default=10)
    top_p: float = field(default=0.9)
    temperature: float = field(default=1.0)

    gen_batch_size: int = field(default=45)
    max_length: int = field(default=None)
    min_length: int = field(default=None)
    do_sample: bool = field(default=True)
    num_beams: int = field(default=1)

    max_iter: int=field(default=1000)
    initial_sample: int=field(default=20)
    schedule: str = field(default="0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0")
    sample_size: str = field(default="0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25")

    

