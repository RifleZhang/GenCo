gpu_ids=${1:-0}
data_name=${2:-agnews}
num_per_class=${3:-800}
exp_name=${4:-t0.8l64}
seed=100
data_name=${data_name}-${num_per_class}-${seed}

export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

# model_name=princeton-nlp/sup-simcse-roberta-large
# model_name=facebook/contriever
model_name=princeton-nlp/sup-simcse-roberta-base
gen_model_name=chavinlo/alpaca-native
#model_name=princeton-nlp/sup-simcse-bert-base-uncased
model_name_short=${model_name##*/}

DATA_DIR=$BASE_DIR/data
TEXT_DATA_DIR=$DATA_DIR/text/sample_data
#GEN_DATA_DIR=$DATA_DIR/gen_data


# input
# corpus_path=$TEXT_DATA_DIR/$data_name/${split}.txt
# context_path=$GEN_DATA_DIR/$data_name/gpt-j-6B/context_with_prompt.jsonl
echo $corpus_path

data_dir=$TEXT_DATA_DIR/$data_name
cache_dir=$BASE_DIR/cache
#data_dir=/usr1/data/ruohongz/classification/data/text/other_data/ag_news_csv

# output
save_dir=$BASE_DIR/save/$data_name
output_dir=$save_dir

mkdir -p $output_dir

# default
initial_sample=40
schedule="0.1,0.2,0.3,0.4,0.6,0.8,1.0"
sample_size="0.075,0.1,0.125,0.16,0.2,0.25,0.3"

eval_batch_size=500
python -m genco.experiment.cond_aug_selflearn \
    --exp_name $exp_name --seed $seed \
    --data_dir $data_dir --output_dir $output_dir --cache_dir $cache_dir \
    --max_iter 2000 \
    --initial_sample $initial_sample --schedule $schedule --sample_size $sample_size \
    --model_name_or_path $model_name --gen_model_name $gen_model_name \
    --per_device_eval_batch_size $eval_batch_size \
    --learning_rate 1e-5 \
    --max_text_length 128 \
    --dataloader_num_workers 1 


