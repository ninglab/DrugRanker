#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : prepare.py
# Author            : Vishal Dey <dey.78@osu.edu>
# Date              : Thu 01 Sep 2022 12:41:55
# Last Modified Date: Thu 01 Sep 2022 12:41:55
# Last Modified By  : Vishal Dey <dey.78@osu.edu>


"""
Prepare and reformat the data for CCLERank code:
- Inputs: AUC list file, CV split indices file
- Outputs: training and testing cell line -- drug response matrix for each fold
"""
import os
import sys
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser

def get_LRO_data(data_dir):
    set_clids = set()
    set_drugs = set()

    sep = '\t' #if 'prism' in data_dir else ','
    auc = {}
    # store {(cell, cpd) = auc}
    with open(data_dir+'aucs.txt', 'r') as fp:
        #next(fp)
        for line in fp.readlines():
            tmp = line.strip().split(sep)
            set_clids.add(tmp[0])
            set_drugs.add(tmp[1])
            auc[(tmp[0], tmp[1])] = float(tmp[2])

    #set_clids = np.genfromtxt(data_dir+'cells.txt', delimiter='\n', dtype=str)
    # store train and test cpd ids per fold per cell line
    indices_pc = {}
    for k in range(5):
        indices_pc[k] = dict()
        indices_pc[k]['train'] = dict.fromkeys(set_clids)
        indices_pc[k]['val']   = dict.fromkeys(set_clids)
        indices_pc[k]['test']  = dict.fromkeys(set_clids)

    sep = '|'
    with open(data_dir+'splits.txt', 'r') as fp:
        for line in fp.readlines():
            tmp = line.strip().split('\t')
            if tmp[0] not in set_clids:
                continue
            indices_pc[int(tmp[1])]['train'][tmp[0]] = tmp[2].split(sep)
            indices_pc[int(tmp[1])]['val'][tmp[0]] = tmp[3].split(sep)
            indices_pc[int(tmp[1])]['test'][tmp[0]] = tmp[4].split(sep)
    return set_clids, set_drugs, auc, indices_pc


def prepare_data(args):
    all_cells, all_drugs, auc, indices = get_LRO_data(args.data_dir)
    #all_drugs = set([_[1] for _ in auc.keys()])
    #all_drugs = np.genfromtxt(args.data_dir + 'drugs.txt', delimiter='\n', dtype=str)
    all_cells, all_drugs = list(all_cells), list(all_drugs)
    print(len(all_cells), len(all_drugs))

    count_missing = 0
    matrix = np.zeros((len(all_cells), len(all_drugs)))

    for i, cell in enumerate(all_cells):
        for j, drug in enumerate(all_drugs):
            try:
                matrix[i][j] = str(auc[(cell, drug)])
            except:
                matrix[i][j] = 'nan'
                count_missing += 1

    #prec = '%.15f' if 'prism' in args.data_dir else '%.4f'
    np.savetxt(args.data_dir+'auc_matrix.txt', matrix, fmt='%s', delimiter=',')
    np.savetxt(args.data_dir+'cells.txt', np.asarray(all_cells), fmt='%s', delimiter='\n')
    np.savetxt(args.data_dir+'drugs.txt', np.asarray(all_drugs), fmt='%s', delimiter='\n')
    prepare_cv_data(matrix, all_cells, all_drugs, indices, args.save_dir)


def prepare_cv_data(auc_matrix, all_cells, all_drugs, indices, data_dir, percentile=5):
    """ Creates separate train and test matrices, threshold file for each split """
    for fold in range(5):
        os.makedirs(data_dir + f'fold_{fold}') if not os.path.exists(data_dir + f'fold_{fold}') else None

        train_matrix = auc_matrix.copy()
        val_matrix   = auc_matrix.copy()
        test_matrix  = auc_matrix.copy()
        thresh = []

        for i, cell in enumerate(all_cells):
            train_drugs = indices[fold]['train'][cell]
            val_drugs   = indices[fold]['val'][cell]
            test_drugs  = indices[fold]['test'][cell]
            train_idx   = np.in1d(all_drugs, train_drugs).nonzero()[0]
            val_idx     = np.in1d(all_drugs, val_drugs).nonzero()[0]
            test_idx    = np.in1d(all_drugs, test_drugs).nonzero()[0]

            train_matrix[i][test_idx] = train_matrix[i][val_idx] = 0
            val_matrix[i][train_idx] = val_matrix[i][test_idx] = 0
            test_matrix[i][train_idx] = test_matrix[i][val_idx] = 0

            # compute threshold from training matrix excluding non-zero and nan values
            mask = ~np.isnan(train_matrix[i]) & (train_matrix[i]!=0)
            thresh.append(np.nanpercentile(train_matrix[i, mask], percentile, method='closest_observation'))

        #prec = '%.15f' if 'prism' in args.data_dir else '%.4f'
        np.savetxt(args.save_dir+f'fold_{fold}/train_matrix.txt', train_matrix, fmt='%s', delimiter=',')
        np.savetxt(args.save_dir+f'fold_{fold}/val_matrix.txt', val_matrix, fmt='%s', delimiter=',')
        np.savetxt(args.save_dir+f'fold_{fold}/test_matrix.txt', test_matrix, fmt='%s', delimiter=',')
        np.savetxt(args.save_dir+f'fold_{fold}/rel.txt', np.asarray(thresh), fmt='%s', delimiter='\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', help='Path to the directory containing the data.')
    parser.add_argument('--save_dir', help='Path to the directory to save files.')
    parser.set_defaults(data_dir='ctrpv2/LRO/', save_dir='ctrpv2/LRO/pletorg/')
    args = parser.parse_args()
    prepare_data(args)
