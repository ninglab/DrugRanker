#!/bin/bash

log_dir='/fs/ess/scratch/PCON0041/Vishal/expts/rank/' # change this to the path where the results will be saved

outd=(25 50 100)
outd_ae=(128)
alpha=(0.1 0.5 1)

model="pairpushc" # Pair-PushC in the paper
fgen="morgan_count"
bs=64
score="linear"
dataset="ctrp"
setup=2
del=5
gs=1

ae_expt_dir="/fs/ess/PCON0041/Vishal/DrugRank/expts/ae/LCO/${dataset}/"  # change this to the path where the AE models are saved

for aed in ${outd_ae[@]}; do
	for mold in ${outd[@]}; do
		for al in ${alpha[@]}; do
			expt_dir="${log_dir}/${dataset}/tmp/setup${setup}/del_${del}/${model}/${fgen}/"
			save_dir="${expt_dir}/${mold}_2_${al}_${score}/"
			mkdir -p $save_dir ${expt_dir}/slurm/ 
			
			for fold in $(seq 0 4); do
				sbatch -A PCON0041 --output=${expt_dir}/slurm/%j.log scripts/evaluate.pbs $model $mold $bs ${ae_expt_dir} $aed $fgen $save_dir $del $score 2 $fold $gs $dataset $setup 100 $al
			done
		done
	done
done
