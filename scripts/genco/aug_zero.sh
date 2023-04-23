gpu_ids=${1:-0}
data_name=${2:-agnews}
num_per_class=${3:-200}
exp_name=${4:-t0.8}
max_text_length=${5:-128}
seed=${6:-100}

data_name=${data_name}-${num_per_class}-${seed}

export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

# model_name=princeton-nlp/sup-simcse-roberta-large
# model_name=facebook/contriever
model_name=princeton-nlp/sup-simcse-roberta-base
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

eval_batch_size=500
batch_size=32
# rand=$RANDOM
# port=$((19000 + $rand % 1000))
# python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $port \
python -m genco.experiment.aug_zero \
    --exp_name seed100 --seed $seed \
    --data_dir $data_dir --output_dir $output_dir --cache_dir $cache_dir \
    --model_name_or_path $model_name \
    --model_name_short $model_name_short \
    --per_device_eval_batch_size $eval_batch_size  \
    --per_device_train_batch_size $batch_size  \
    --exp_name $exp_name \
    --max_text_length $max_text_length \
    --dataloader_num_workers 1


