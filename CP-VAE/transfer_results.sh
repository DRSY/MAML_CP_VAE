for s in 1 2
do
    for t in {1..7}
    do
        echo "Transfering s${s}_t${t} to val folder"
        dir=./checkpoint/ours-s${s}_t${t}-glove
        if [[ $s -eq 1 ]]
        then
            last_ckpt=$(ls -lt $dir | sed -n '6p' | awk '{print $9}')
            echo "s1"
        else
            last_ckpt=$(ls -lt $dir | sed -n '5p' | awk '{print $9}')
            echo "s2"
        fi
        head -n 1000 ./checkpoint/ours-s${s}_t${t}-glove/${last_ckpt}/generated_results.txt > /home/roy/style_transfer/cnn_cls_modified/val/maml_cp_vae/s${s}/t${t}.1
        tail -n 1000 ./checkpoint/ours-s${s}_t${t}-glove/${last_ckpt}/generated_results.txt > /home/roy/style_transfer/cnn_cls_modified/val/maml_cp_vae/s${s}/t${t}.0
        cat /home/roy/style_transfer/cnn_cls_modified/val/maml_cp_vae/s${s}/t${t}.0 /home/roy/style_transfer/cnn_cls_modified/val/maml_cp_vae/s${s}/t${t}.1 > /home/roy/style_transfer/cnn_cls_modified/val/maml_cp_vae/s${s}/t${t}.val
    done
done
echo "all done"