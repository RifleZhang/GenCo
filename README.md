# GenCo

Note Alpaca requires tranformers>='4.28.0.dev0'
Setup Environment (for reference)
```
# conda env
conda env create -f environment.yml
pip install -r requirement.txt
conda activate cls

# correctly config path
export BASE_DIR=/usr1/data/ruohongz/classification
export CODE_DIR=/usr0/home/ruohongz/classification/GenCo

# install GenCo package
pip install -e .
```


# download data

# sample data
```
export data_name=agnews
bash scripts/prep/sample_data.sh agnews 800 2000
bash scripts/gen/generate_context.sh 4,5,6,7 agnews 800 t0.8l64 chavinlo/alpaca-native 0.8 0.95
bash scripts/ginco/aug_zero.sh 0 agnews 800 t0.8l64 128
bash scripts/gen/cond_gen_context.sh 4,5,6,7 agnews 800
```

