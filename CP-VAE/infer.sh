for s in 1 2
do
    for t in {1..7}
    do
        dir=./checkpoint/ours-s${s}_t${t}-glove
        last_ckpt=$(ls -lt $dir | sed -n '2p' | awk '{print $9}')
        if [[ -e $dir/${last_ckpt}/model.pt ]]
        then
            echo "start transfer using $dir/${last_ckpt}/model.pt"
            python transfer.py --data_name s${s}_t${t} --load_path $dir/$last_ckpt
        else
            echo "model.pt does not exist"
        fi
    done
done