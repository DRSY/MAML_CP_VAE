# MAML_CP_VAE
This repo contains Pytorch implementation of the paper ST2:Small-data Text Style Transfer via Multi-task Meta-Learning. Up to now only CP-VAE version is contained, CrossAlign and VAE versions will be 
updated soom.

## Dependencies
- torch==1.3.1
- python > 3.6
- tqdm
- pandas
- numpy
- scipy

## Usage
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