#!/bin/usr/bash/env bash
corpus=$1
num_tasks=$2

mkdir ../output ../ckpt ../emb
mkdir ../output/$corpus ../ckpt/$corpus ../data/$corpus/processed ../data/$corpus/processed/${num_tasks}t ../emb/$corpus
for ((t=1; t<=num_tasks; t++)); do
	mkdir ../emb/$corpus/t$t
done
