#!/bin/bash

outd=(25 50 100)
gsteps=(4 16 32 64)
model="listone" # List-One in the paper
bs=1
dataset=("ctrp" "prism")
setups=("LRO" "LCO")
score="linear"

for setup in ${setups[@]}; do
	for data in ${dataset[@]}; do
		log_dir="/fs/ess/scratch/PCON0041/Vishal/DrugRank/expts/rank/${data}/20230902-${setup}/del_5/" # change this to the path where the results will be saved
		for mold in ${outd[@]}; do
			for gs in ${gsteps[@]}; do
				expt_dir="${log_dir}/${model}/"
				save_dir="${expt_dir}/${mold}_2_${gs}_${score}/"
				mkdir -p $save_dir ${expt_dir}/slurm/ 
			
				sbatch -A PCON0041 --output=${expt_dir}/slurm/%j.log scripts/evaluate.pbs \
				$model $mold $bs $save_dir -1 $gs $data $setup 100
				#bash scripts/evaluate.pbs $model $mold $bs $save_dir -1 $gs $data $setup 100
			done
		done
	done
done
