DATA_DIR=$BASE_DIR/data
CUSTOM_DATA_DIR=$DATA_DIR/text/other_data
SAMPLE_DATA_DIR=$DATA_DIR/text/sample_data

data_name=${1:-agnews}
num_per_class=${2:-500}
eval_num_per_class=${3:-2000}
seed=${4:-100}

inp_dir=$CUSTOM_DATA_DIR/$data_name
out_dir=$SAMPLE_DATA_DIR/${data_name}-${num_per_class}-${seed}
mkdir -p $out_dir

# out: $out_dir/label-{class or name}.jsonl
python -m genco.processor.sample_data \
    --data_dir $inp_dir --data_prefix train \
    --save_dir $out_dir --num_per_class $num_per_class --seed $seed

python -m genco.processor.sample_data \
    --data_dir $inp_dir --data_prefix test \
    --save_dir $out_dir --num_per_class $eval_num_per_class --seed $seed

cp $inp_dir/crafted_label_names.txt $out_dir/crafted_label_names.txt
#cp $inp_dir/label_names.txt $out_dir/label_names.txt
#cp $inp_dir/prompt.txt $out_dir/prompt.txt
#cp $inp_dir/*.json $out_dir/