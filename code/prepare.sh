for s in 1 2
do
    for t in {1..7}
    do
        cat ./data/s${s}/train/t${t}.0 ./data/s${s}/train/t${t}.1 > ./data/s${s}/train/t${t}.all
        cat ./data/s${s}/val/t${t}.0 ./data/s${s}/val/t${t}.1 > ./data/s${s}/val/t${t}.all
    done
done
echo "done"