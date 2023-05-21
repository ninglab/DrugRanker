#!/bin/bash

log_dir='/fs/ess/scratch/PCON0041/Vishal/expts/rank/' # change this to the path where the results will be saved

outd=(25 50 100)
gsteps=(4 16 64)
outd_ae=(128)

model="listall" # List-All in the paper
fgen="morgan_count"
bs=1
score="linear"
dataset="ctrp"
setup=2
del=5

ae_expt_dir="/fs/ess/PCON0041/Vishal/DrugRank/expts/ae/20221102/${dataset}/"  # change this to the path where the AE models are saved

for aed in ${outd_ae[@]}; do
	for mold in ${outd[@]}; do
		for gs in ${gsteps[@]}; do
			expt_dir="${log_dir}/${dataset}/tmp/setup${setup}/del_${del}/${model}/${fgen}/"
			save_dir="${expt_dir}/${mold}_2_${gs}_${score}/"
			mkdir -p $save_dir ${expt_dir}/slurm/ 
			
			for fold in $(seq 0 0); do
				sbatch -A PCON0041 --output=${expt_dir}/slurm/%j.log scripts/evaluate.pbs $model $mold $bs ${ae_expt_dir} $aed $fgen $save_dir $del $score 2 $fold $gs $dataset $setup
			done
		done
	done
done
