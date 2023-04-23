gpu_ids=${1:-0}
data_name=${2:-agnews}
num_per_class=${3:-200}
exp_name=${4:-temp}
model_name=${5:-chavinlo/alpaca-native}
temperature=${6:-0.8}
top_p=${7:-0.95}
seed=${8:-100}

data_name=${data_name}-${num_per_class}-${seed}

export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

# source env.sh to set up the environment path
#BASE_DIR=/usr0/home/ruohongz/classification
#BASE_DIR=/home/ruohongzhang/classification

#model_name=princeton-nlp/sup-simcse-bert-base-uncased
#model_name=gpt2
#model_name=gpt2-xl
#model_name=EleutherAI/gpt-j-6B
#model_name=chavinlo/alpaca-native
#model_name=decapoda-research/llama-7b-hf
model_name_short=${model_name##*/}

# input
DATA_DIR=$BASE_DIR/data
CACHE_DIR=$BASE_DIR/cache
TEXT_DATA_DIR=$DATA_DIR/text/sample_data
SAVE_DIR=$BASE_DIR/save

# output
# GEN_DATA_DIR=$DATA_DIR/gen_data/$data_name
gen_opts="--top_p $top_p --temperature $temperature"
#gen_opts="--top_p 0.9 --temperature 0.9" # --repetition_penalty 1.2"
#gen_opts="--top_p 0.8 --temperature 0.95"
#max_length=96
#min_length=32
max_length=128
min_length=64
num_samples=5

data_dir=$TEXT_DATA_DIR/$data_name
cache_dir=$BASE_DIR/cache
save_dir=$BASE_DIR/save/$data_name
output_dir=$save_dir

# input
batch_size=32

rand=$RANDOM
port=$((19000 + $rand % 1000))
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $port \
    -m genco.experiment.generate_context \
    --data_dir $data_dir --output_dir $output_dir --cache_dir $cache_dir \
    --gen_model_name $model_name --model_name_or_path $model_name \
    --num_samples $num_samples --per_device_eval_batch_size $batch_size \
    --dataloader_num_workers 1 --seed $seed --fp16 \
    --max_length $max_length --min_length $min_length \
    --exp_name $exp_name \
    $gen_opts

# save_path=$SAVE_DIR/gen_train_text.jsonl
# data_path=$TEXT_DATA_DIR/$data_name/train.txt

# # input
# batch_size=4
# python -m xmclib.driver.generate_context \
#     --data_path $data_path --save_path $save_path --output_dir $SAVE_DIR \
#     --model_name_or_path $model_name --cache_dir $CACHE_DIR \
#     --num_samples $num_samples --per_device_eval_batch_size $batch_size \
#     --dataloader_num_workers 1 --seed $seed --fp16 \
#     --max_length $max_length --min_length $min_length \
#     $gen_opts
