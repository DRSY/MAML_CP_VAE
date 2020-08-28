#!/bin/usr/bash/env bash

for corpus in s1 s2
do
	echo ">>>>>>> $corpus <<<<<<<"
	python3 main.py --corpus $corpus --maml-epochs 20 --transfer-epochs 10 --epochs-per-val 5 --config-path ../config/${corpus}.json --maml-batch-size 64 --sub-batch-size 128 --train-batch-size 128
done
