s=$1
t=$2

# make text.pretrain, t1.all, t2.all, etc.
bash scripts/get_pretrain_text.sh s$s

# make dirs (with 7 tasks)
bash scripts/make_dirs.sh s$s 7

# --------------------------------
# run original vae

# train word2vec model
python3 wdv.py --config-path ../config/s$s.json --corpus s$s --task-id $t --model vae

# test original models
# :initial
python3 original.py --config-path ../config/s$s.json --corpus s$s --task-id $t --batch-size 64 --epochs 20 --epochs-per-val 5 --pretrain-epochs 5
# :load last ckpt / load processed data (from saved config)
python3 original.py --config-path ../output/s$s/t$t.json --corpus s$s --task-id $t --batch-size 64 --epochs 20 --epochs-per-val 5 --pretrain-epochs 5 --load-model --load-data

# run inference (and dump embeddings)
python3 original.py --config-path ../output/s$s/t$t.json --corpus s$s --task-id $t --inference --dump-embeddings --from-pretrain # (if pretrain phase exists)

# --------------------------------
# run maml_vae

# train word2vec model
python3 wdv.py --config-path ../config/s$s.json --corpus s$s --task-id 0 --model maml

# test maml model
# :initial 
python3 main.py --corpus s$s --maml-epochs 20 --transfer-epochs 10 --epochs-per-val 5 --config-path ../config/s$s.json --maml-batch-size 16 --sub-batch-size 64 --train-batch-size 64
# :load last ckpt / load processed data
python3 main.py --corpus s$s --maml-epochs 20 --transfer-epochs 10 --epochs-per-val 5 --config-path ../config/s$s.json --maml-batch-size 16 --sub-batch-size 64 --train-batch-size 64 --load-model --load-data

# run inference (and dump embeddings)
python3 main.py --config-path ../config/s$s.json --corpus s$s --task-id $t --dump-embeddings

# extract embeddings
python3 main.py --config-path ../config/s$s.json --corpus s$s --extract-embeddings --task-id $t --ckpt epoch-1.t$t # if from pretrain to load vocab: --from-pretrain

# online-inference
python3 main.py --online-inference --config-path ../config/s$s.json --corpus s$s --ckpt epoch-1.t$t --tgt-file ../data/s$s/val/t$t.0 # /1


# DONE!



