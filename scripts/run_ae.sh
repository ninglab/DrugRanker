#!/bin/bash

outdim=(32 64 128 256)
batch=(16 32 64)
data=$1
expt_dir="/fs/ess/PCON0041/Vishal/DrugRank/expts/ae/20221102/${data}/"
fold=0

# separate pretrained model for each fold
for fold in $(seq 0 4); do
for bs in ${batch[@]}; do
	for outd in ${outdim[@]}; do
		save_dir="${expt_dir}/all_bs_${bs}_outd_${outd}/fold_${fold}/"
		mkdir -p $save_dir
		python src/train_ae.py --genexp_file data/CCLE/CCLE_expression.csv --splits_path data/${data}/LCO/fold_${fold}/ --save_path $save_dir --ae_out_size $outd --bs $bs --cuda > $save_dir/train.log
	done
done
done
