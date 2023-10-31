#!/bin/bash

outd=(25 50 100)
alpha=(0.1 0.5 1)

model="pairpushc" # Pair-PushC in the paper
bs=32
dataset=("ctrp" "prism")
setups=("LRO" "LCO")
score="linear"

gs=1

for setup in ${setups[@]}; do
	for data in ${dataset[@]}; do
		log_dir="/fs/ess/scratch/PCON0041/Vishal/DrugRank/expts/rank/${data}/20230902-${setup}/del_5/"
		for mold in ${outd[@]}; do
			for al in ${alpha[@]}; do
				expt_dir="${log_dir}/${model}/"
				save_dir="${expt_dir}/${mold}_2_${al}_${score}/"
				mkdir -p $save_dir ${expt_dir}/slurm/ 
			
				sbatch -A PCON0041 --output=${expt_dir}/slurm/%j.log scripts/evaluate.pbs \
				$model $mold $bs $save_dir $fold $gs $data $setup 100 $al
				#bash scripts/evaluate.pbs $model $mold $bs $save_dir $fold $gs $data $setup 100 $al
			done
		done
	done
done
