# GenCo

This is the official code for paper "Generation-driven Contrastive Self-training for Zero-shot Text Classification with Instruction-tuned GPT"

# Setup Environment (for reference)
Note Alpaca requires tranformers>='4.28.0.dev0', we used transformers 4.28.0.dev0 by cloning from the transformers github. 

```
# conda env (or pip install whatever needed)
conda env create -f environment.yml
pip install -r requirement.txt
conda activate cls

# config path
export BASE_DIR=/path/to/store/data_and_result
export CODE_DIR=/code_dir

# install GenCo package
pip install -e .
```


# Prepare data
download agnews, dbpedia and amazon (optionally yelp, imbd) from https://github.com/yumeng5/LOTClass

Yahoo answers from https://www.kaggle.com/datasets/soumikrakshit/yahoo-answers-dataset

Our label names are in folder data_prompt.


# Reference scripts for experiments
We provide reference scripts for datasets with multiple classes, which may require more hyperparam tuning compared with binary sentiment classification (Yelp, IMDB or Amazon).

agnews
```
export data_name=agnews
# prepare and sample data (we only random sample a small subset of training)
bash scripts/prep/sample_data.sh $data_name 1000 2000
# generate augmented text by Alpaca (multi gpus for faster inference)
bash scripts/gen/generate_context.sh 0,1,2,3 $data_name 1000 t0.8ins chavinlo/alpaca-native 0.8 0.95
# predict pseudo label
bash scripts/ginco/aug_zero.sh 0 $data_name 1000 t0.8ins 128
# conditional generation for pseudo labels (can skip)
bash scripts/gen/cond_gen_context.sh 0,1,2,3 $data_name 1000
# contrastive self-train (gpu 0 for self-train BERT, gpu 1 for inference Alpaca)
bash scripts/genco/cond_aug_selflearn.sh 0,1 $data_name 1000 t0.8ins
```

dbpedia
```
export data_name=dbpedia
bash scripts/prep/sample_data.sh $data_name 800 2000
bash scripts/gen/generate_context.sh 0,1,2,3 $data_name 800 t0.8ins chavinlo/alpaca-native 0.8 0.95
bash scripts/ginco/aug_zero.sh 0 $data_name 800 t0.8ins 128
(optionally)
bash scripts/gen/cond_gen_context.sh 0,1,2,3 $data_name 800
# contrastive self-train (gpu 0 for self-train BERT, gpu 1 for inference Alpaca)
bash scripts/genco/cond_aug_selflearn.sh 0,1 $data_name 800 t0.8ins
```

yahoo answer
```
export data_name=yahoo_answers_csv
bash scripts/prep/sample_data.sh $data_name 1500 2000
bash scripts/gen/generate_context.sh 0,1,2,3 $data_name 1500 t0.8ins chavinlo/alpaca-native 0.8 0.95
bash scripts/ginco/aug_zero.sh 0 $data_name 1500 t0.8ins 128
(optionally)
bash scripts/gen/cond_gen_context.sh 0,1,2,3 $data_name 1500
# contrastive self-train (gpu 0 for self-train BERT, gpu 1 for inference Alpaca)
bash scripts/genco/cond_aug_selflearn.sh 0,1 $data_name 1500 t0.8ins
```

# Citation
To be added
