for s in 1 2
do
	for t in 1 2 3 4 5 6 7
	do
		python3 original.py --config-path ../config/s$s.json --corpus s$s --task-id $t --batch-size 64 --epochs 20 --epochs-per-val 5 --pretrain-epochs 0
	done
done
