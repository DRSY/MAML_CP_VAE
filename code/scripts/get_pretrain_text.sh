corpus=$1
dir=../data/$corpus

for t in 1 2 3 4 5 6 7
do
	cat $dir/train/t$t.0 $dir/train/t$t.1 >> $dir/text.pretrain
	cat $dir/train/t$t.0 $dir/train/t$t.1 > $dir/t$t.all
done
