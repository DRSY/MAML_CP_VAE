for s in 1 2
do
	for t in 1 2 3 4 5 6 7
	do
		python3 main.py --config-path ../config/s$s.json --corpus s$s --maml-epochs 0 --transfer-epochs 0 --infer-task-id $t
	done
done
