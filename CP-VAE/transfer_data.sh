for s in 1 2
do
    for t in {1..7}
    do
        mkdir -p ./data/s${s}_t${t}
        cp /home/roy/style_transfer/style/maml/maml-cross-align/data/s${s}/train/t${t}.0 ./data/s${s}_t${t}/sentiment.train.0
        cp /home/roy/style_transfer/style/maml/maml-cross-align/data/s${s}/train/t${t}.1 ./data/s${s}_t${t}/sentiment.train.1
        cp /home/roy/style_transfer/style/maml/maml-cross-align/data/s${s}/val/t${t}.0 ./data/s${s}_t${t}/sentiment.dev.0
        cp /home/roy/style_transfer/style/maml/maml-cross-align/data/s${s}/val/t${t}.1 ./data/s${s}_t${t}/sentiment.dev.1
        cp /home/roy/style_transfer/style/maml/maml-cross-align/data/s${s}/val/t${t}.0 ./data/s${s}_t${t}/sentiment.test.0
        cp /home/roy/style_transfer/style/maml/maml-cross-align/data/s${s}/val/t${t}.1 ./data/s${s}_t${t}/sentiment.test.1
    done
done