#!/bin/bash

outdim=(32 64 128 256)
batch=(16 32 64)
data=$1
expt_dir="/fs/ess/PCON0041/Vishal/DrugRank/expts/ae/LCO/${data}/"
fold=0

# separate pretrained model for each fold in LCO experiments
for fold in $(seq 0 4); do
for bs in ${batch[@]}; do
	for outd in ${outdim[@]}; do
		save_dir="${expt_dir}/all_bs_${bs}_outd_${outd}/fold_${fold}/"
		mkdir -p $save_dir
		python src/train_ae.py --genexp_file data/CCLE/CCLE_expression.csv --splits_path data/${data}/LCO/splits/fold_${fold}/ \
		--save_path $save_dir --ae_out_size $outd --bs $bs --cuda > $save_dir/train.log
	done
done
done


# to pretrain single GeneAE model using all cell lines in LRO experiments
expt_dir="/fs/ess/PCON0041/Vishal/DrugRank/expts/ae/LRO/${data}/"
save_dir="${expt_dir}/all_bs_${bs}_outd_${outd}/fold_${fold}/"
mkdir -p $save_dir
python src/train_ae.py --genexp_file data/CCLE/CCLE_expression.csv --save_path $save_dir --ae_out_size $outd --bs $bs --use_all --cuda > $save_dir/train.log