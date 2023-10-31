#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : similarity.py
# Author            : Vishal Dey <dey.78@osu.edu>
# Date              : Mon 21 Feb 2022 22:05:59
# Last Modified Date: Mon 21 Feb 2022 23:26:06
# Last Modified By  : Vishal Dey <dey.78@osu.edu>

import pdb
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from itertools import product

def read_smiles():
	""" return comp_ids: (smiles, morgan fingerprint) """
	smiles = {}

	with open('cmpd_id_name_group_smiles.txt', 'r') as fp:
		next(fp)
		for line in fp.readlines():
			tmp = line.strip().split('\t')
			smiles[tmp[0]] = (tmp[-1], AllChem.GetMorganFingerprint(Chem.MolFromSmiles(tmp[-1]), radius=3))
	return smiles


def compute_tanimoto(mols1, mols2):
	similarities = []

	ind = []
	for i, j in product(range(len(mols1)), range(len(mols2))):
		if i <= j and mols1[i] != mols2[j]:
			ind.append((i, j))

	for i, j in ind:
		similarity = DataStructs.TanimotoSimilarity(mols1[i], mols2[j])
		similarities.append(similarity)

	similarities = np.array(similarities)
	return np.mean(similarities)
	#print(f'{np.mean(similarities):.4f}', end=',')


def compute_similarity(sens, insens):
	return compute_tanimoto(sens, sens), compute_tanimoto(insens, insens), \
		compute_tanimoto(sens, insens)

def compute_threshold(sens, percentile=5):
	return np.nanpercentile(sens, percentile)

def group_drugs(drugs, sens_values, thresh):
	sensitive = drugs[np.where(sens_values <= thresh)]
	insensitive = drugs[np.where(sens_values > thresh)]
	#print(thresh, len(sensitive), len(insensitive))

	return thresh, sensitive, insensitive


def get_cv_data(auc_file, splits_file):
	set_clids = set()

	auc = {}
	# store {(cell, cpd) = auc}
	with open(auc_file, 'r') as fp:
		next(fp)
		for line in fp.readlines():
			tmp = line.strip().split(',')
			set_clids.add(tmp[0])
			auc[(tmp[0], tmp[1])] = float(tmp[2])

	# store train and test cpd ids per fold per cell line
	indices_pc = {}
	for k in range(5):
		indices_pc[k] = dict()
		indices_pc[k]['train'] = dict.fromkeys(set_clids)
		indices_pc[k]['test']  = dict.fromkeys(set_clids)


	with open(splits_file, 'r') as fp:
		for line in fp.readlines():
			tmp = line.strip().split('\t')
			if tmp[0] not in set_clids:
				continue
			indices_pc[int(tmp[1])]['train'][tmp[0]] = tmp[2].split(',')
			indices_pc[int(tmp[1])]['test'][tmp[0]] = tmp[3].split(',')
	return set_clids, auc, indices_pc


def main(auc_file='final_list_auc.txt', splits_file='./splits.txt'):
	smiles = read_smiles()
	set_clids, auc, indices_pc = get_cv_data(auc_file, splits_file)

	for k in range(5):
		fout = open(f'analysis/stat/group_stat_{k+1}.txt', 'w')
		fout2 = open(f'analysis/sim/group_sim_{k+1}.txt', 'w')

		print('cell 5%-threshold #sensitive #insensitive #total', file=fout)
		print('cell sens-sens insens-insens sens-insens', file=fout2)
		for cell in set_clids:
			drugs = indices_pc[k]['train'][cell]
			sens = [auc[(cell, d)] for d in drugs]
			thresh = compute_threshold(sens, 5)

			drugs = np.asarray(indices_pc[k]['test'][cell])
			sens = [auc[(cell, d)] for d in drugs]
			thresh, sensitive, insensitive = group_drugs(drugs, sens, thresh)

			## print some statistics
			print(cell, f'{thresh:.3f}', len(sensitive), len(insensitive), len(sensitive)+len(insensitive), file=fout)

			## compute similarity
			sens = [smiles[_][1] for _ in sensitive]
			insens = [smiles[_][1] for _ in insensitive]
			a, b, c = compute_similarity(sens, insens)
			print(cell, f'{a:.3f} {b:.3f} {c:.3f}', file=fout2)

		fout.close()
		fout2.close()

if __name__ == '__main__':
	main()
