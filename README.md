# MAML_CP_VAE
This repository contains Pytorch implementation of the paper ST2:Small-data Text Style Transfer via Multi-task Meta-Learning. Up to now only CP-VAE version is contained, CrossAlign and VAE versions will be 
updated soon.

## Dependencies
- torch==1.3.1
- python > 3.6
- tqdm
- pandas
- numpy
- scipy

## Usage
clone the repo and install required packages
```bash
git clone https://github.com/DRSY/MAML_CP_VAE.git
pip install -r requirements.txt
```
enter the code dir
```bash
cd maml-cp-vae/code
```
generate corpus for building vocab
```bash
bash scripts/get_pretrain_text.sh
```
make dirs
```bash
bash scripts/make_dirs.sh
```
set the corpus(s1 or s2)
```bash
export corpus=s1
export s=1
```
start training and inference
```bash
python main.py --configpath ../config/s$s.json --corpus $corpus --maml-epochs 20 --transfer-epochs 10 --epochs-per-val 5 --maml-batch-size 8 --sub-batch-size --train-batch-size 16 --device-idx 0
```

inference after task-specific fine-tuning
```bash
bash scripts/infer_tsf.sh
```

## Acknowledgement
Underpinning code for cp-vae is adapted from [CP-VAE](https://github.com/BorealisAI/CP-VAE).