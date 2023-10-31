import csv
import os
import pandas as pd

import numpy as np
from scipy import stats
from collections import defaultdict


def get_auc(auc_file):
    set_clids = set()

    sep = '\t' if ('prism' in auc_file or 'ctrpv2' in auc_file) else ','
    auc = {}
    # store {(cell, cpd) = auc}
    with open(auc_file, 'r') as fp:
        #next(fp)
        for line in fp.readlines():
            tmp = line.strip().split(sep)
            set_clids.add(tmp[0])
            auc[(tmp[0], tmp[1])] = tmp[2]
    return auc, set_clids


def get_cv_data_LRO(auc_file, splits_dir):

    auc, set_clids = get_auc(auc_file)
    set_clids = np.genfromtxt(os.path.join(splits_dir, 'cells.txt'), delimiter='\n', dtype=str)
    threshold_pc = {}
    # store train and test cpd ids per fold per cell line
    indices_pc = {}
    for k in range(5):
        indices_pc[k] = dict()
        indices_pc[k]['train'] = dict.fromkeys(set_clids)
        indices_pc[k]['val'] = dict.fromkeys(set_clids)
        indices_pc[k]['test']  = dict.fromkeys(set_clids)
        thresholds = np.genfromtxt(os.path.join(splits_dir, f'pletorg/fold_{k}/rel.txt'), delimiter='\n')
        threshold_pc[k] = dict(zip(set_clids, thresholds))

    sep = '|'
    with open(os.path.join(splits_dir, 'splits.txt'), 'r') as fp:
        for line in fp.readlines():
            tmp = line.strip().split('\t')
            if tmp[0] not in set_clids:
                continue
            indices_pc[int(tmp[1])]['train'][tmp[0]] = tmp[2].split(sep)
            indices_pc[int(tmp[1])]['val'][tmp[0]] = tmp[3].split(sep)
            indices_pc[int(tmp[1])]['test'][tmp[0]] = tmp[4].split(sep)
    return auc, indices_pc, threshold_pc


def get_cv_data_LCO(auc_file, splits_dir):
    auc, set_clids = get_auc(auc_file)
    # store train and test cpd ids per fold per cell line
    indices_pc = {}
    threshold_pc = {}
    for k in range(5):
        indices_pc[k] = dict()
        threshold_pc[k] = dict()
        indices_pc[k]['train'] = np.genfromtxt(splits_dir+f'fold_{k}/train_cells.txt', delimiter='\n', dtype=str)
        indices_pc[k]['val'] = np.genfromtxt(splits_dir+f'fold_{k}/val_cells.txt', delimiter='\n', dtype=str)
        indices_pc[k]['test'] = np.genfromtxt(splits_dir+f'fold_{k}/test_cells.txt', delimiter='\n', dtype=str)
        threshold = np.genfromtxt(splits_dir+f'fold_{k}/train_rel.txt')
        threshold_pc[k]['train'] = dict(zip(indices_pc[k]['train'].tolist(), threshold.tolist()))
        threshold = np.genfromtxt(splits_dir+f'fold_{k}/val_rel.txt')
        threshold_pc[k]['val'] = dict(zip(indices_pc[k]['val'].tolist(), threshold.tolist()))
        threshold = np.genfromtxt(splits_dir+f'fold_{k}/test_rel.txt')
        threshold_pc[k]['test'] = dict(zip(indices_pc[k]['test'].tolist(), threshold.tolist()))

    return auc, indices_pc, threshold_pc


def get_fold_data_LRO(auc, train_index, val_index, test_index, smiles_file):
    """ get smiles and auc data from the indices for a particular CV run """
    cpdid_smiles = get_data_from_smiles(smiles_file)

    train_auc, val_auc, test_auc = [], [], []

    for cell, train_cpd in train_index.items():
        train_auc.extend([(cpdid_smiles[cpd], cell, auc[(cell, cpd)], cpd) for cpd in train_cpd])
    
    for cell, val_cpd in val_index.items():
        val_auc.extend([(cpdid_smiles[cpd], cell, auc[(cell, cpd)], cpd) for cpd in val_cpd])

    for cell, test_cpd in test_index.items():
        test_auc.extend([(cpdid_smiles[cpd], cell, auc[(cell, cpd)], cpd) for cpd in test_cpd])

    return train_auc, val_auc, test_auc


def get_fold_data_LCO(auc, train_index, val_index, test_index, smiles_file):
    """ get smiles and auc data from the indices for a particular CV run """
    cpdid_smiles = get_data_from_smiles(smiles_file)

    train_auc, val_auc, test_auc = [], [], []

    for cell in train_index:
        train_auc += [(cpdid_smiles[cpd], cell, auc[(cell, cpd)], cpd) for cpd in cpdid_smiles if (cell,cpd) in auc]

    for cell in val_index:
        val_auc += [(cpdid_smiles[cpd], cell, auc[(cell, cpd)], cpd) for cpd in cpdid_smiles if (cell,cpd) in auc]

    for cell in test_index:
        test_auc += [(cpdid_smiles[cpd], cell, auc[(cell, cpd)], cpd) for cpd in cpdid_smiles if (cell,cpd) in auc]

    return train_auc, val_auc, test_auc

def get_data_from_smiles(smiles_file):
    smiles_dict = {}
    with open(smiles_file, 'r') as fp:
        next(fp)
        for line in fp.readlines():
            tmp = line.strip().split('\t')
            smiles_dict[tmp[0]] = tmp[-1]

    return smiles_dict


def get_thresholds(molecules, delta=5):
    # sensitivity threshold is being computed using only the data in molecules
    auc_list = defaultdict(list)
    threshold = {}
    cell_ids = set()

    for comp in molecules:
        auc_list[comp.clid].append(comp.auc)
        cell_ids.add(comp.clid)

    for clid in cell_ids:
        threshold[clid] = np.nanpercentile(auc_list[clid], delta, interpolation='nearest')
    return threshold


def create_pairs(set1, set2, num_pairs, mixture):
    pairs = []
    for a in set1:
        temp = []
        for b in set2:
            temp.append((a, b))
            if mixture:
                temp.append((b, a))

        if len(temp) > num_pairs:
            ind = np.random.choice(len(temp), num_pairs, replace=False)
            pairs.extend(np.array(temp)[ind])
        else:
            pairs.extend(np.array(temp))
    return pairs


def create_pairs_sameg(drugs, num_pairs):
    pairs = []

    for i in range(len(drugs)):
        temp = []
        for j in range(i+1, len(drugs)):
            temp.append((drugs[i], drugs[j]))
        if len(temp) > num_pairs:
            ind = np.random.choice(len(temp), num_pairs, replace=False)
            pairs.extend(np.array(temp)[ind])
        else:
            pairs.extend(np.array(temp))
    return pairs


def create_pairs_sens_ordered(drugs, num_pairs):
    pairs = []
    for i, comp1 in enumerate(drugs):
        temp = []
        for j, comp2 in enumerate(drugs):
            if comp1.auc < comp2.auc:
                temp.append((comp1, comp2))
                #if mixture:
                #   pairs.append((comp2, comp1))

        if len(temp) > num_pairs:
            ind = np.random.choice(len(temp), num_pairs, replace=False)
            pairs.extend(np.array(temp)[ind])
        else:
            pairs.extend(np.array(temp))
    return np.asarray(pairs)


def create_pairs_random(drugs, num_pairs):
    pairs = []
    set_ind = set()

    for i in range(len(drugs)):
        temp = []
        for j in range(len(drugs)):
            if (j,i) not in set_ind:
                temp.append((i, j))

        if len(temp) > num_pairs:
            ind = np.random.choice(len(temp), num_pairs, replace=False)
            pairs.extend(np.array(temp)[ind])
            for _ in np.array(temp)[ind]:
                set_ind.add(tuple(_))
        else:
            pairs.extend(np.array(temp))
            for _ in temp:
                set_ind.add(_)
    return [(drugs[a], drugs[b]) for a,b in pairs]


def get_pairs(sens, insens, cell_ids, num_pairs, pair_setting, mixture):
    pairs = []
    # create `npair` pairs per each sensitive drug per cell line to include all sensitive ones
    for clid in cell_ids:
        pairs.extend(create_pairs(sens[clid], insens[clid], num_pairs, mixture))

        if pair_setting == 2:
            # order does not matter among pairs from same class
            pairs.extend(create_pairs_sameg(sens[clid], num_pairs))
            pairs.extend(create_pairs_sameg(insens[clid], num_pairs=1))

        elif pair_setting == 3:
            # only create ordered pairs from same class of compounds
            pairs.extend(create_pairs_sens_ordered(sens[clid], num_pairs))
            pairs.extend(create_pairs_sens_ordered(insens[clid], num_pairs=1))
    return np.asarray(pairs)


def get_random_pairs(molecules, num_pairs):
    # get a random subset of pairs -- this may be required during testing
    # when no ground truth AUC is available
    pairs = []
    mols_pc = defaultdict(list)

    # get all mols per cell line
    for m1 in molecules:
        mols_pc[m1.clid].append(m1)

    for cell, mols in mols_pc.items():
        pairs.extend(create_pairs_random(mols, num_pairs))
        #pairs.extend(create_pairs_sameg(mols, num_pairs))
    # during testing, no need to shuffle them since it should not have an effect
    return np.asarray(pairs)


def set_labels(data, threshold):
    for comp in data:
        comp.label = int(comp.auc <= threshold[comp.clid])

