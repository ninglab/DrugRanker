'''
Synopsis: Create a dataloader class that creates training instances with pairs (c_i, c_j)
    such that c_i is ranked above c_j
'''
import numpy as np
import random
from itertools import chain
from collections import defaultdict
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem

from features.features_generators import *
from features.featurization import MolGraph, BatchMolGraph
from dataloader.utils import *

## cache to store molgraphs
SMILES_TO_GRAPH = {}
## cache to store smiles -> mols
SMILES_TO_MOLS = {}

class CellLine:
    def __init__(self, expression_file):
        self.expression = {}
        sep = '\t' if 'Combined' in expression_file else ','
        with open(expression_file, 'r') as fp:
            next(fp)
            for line in fp.readlines():
                tmp = line.strip().split(sep)
                self.expression[tmp[0]] = np.array(tmp[1:], dtype=np.float32) # type: ignore

    def get_expression(self, ccl_ids):
        return [self.expression[_] for _ in ccl_ids]


def get_mol(smiles):
    # caches smiles to RDKit mol
    if smiles in SMILES_TO_MOLS:
        return SMILES_TO_MOLS[smiles]
    SMILES_TO_MOLS[smiles] = Chem.MolFromSmiles(smiles) # type: ignore
    return SMILES_TO_MOLS[smiles]


class MoleculePoint:
    '''
        initialize molecule data point with features
    '''
    def __init__(self, smiles, clid, auc, cpdid, features=None, label=None, feature_gen=None, in_test=True):
        self.smiles = smiles
        self.auc = np.float32(auc) # type: ignore
        self.cpdid = cpdid
        self.label = label # maybe None initially, can be changed later
        self.clid = clid
        self.in_test = in_test
        self.features = features

    def set_feature(self, f):
        self.features = f


def to_molgraph(data):
    # caches molgraphs
    for comp in data:
        if comp.smiles not in SMILES_TO_GRAPH:
            SMILES_TO_GRAPH[comp.smiles] = MolGraph(get_mol(comp.smiles))


class MoleculeDatasetTrain(Dataset):
    def __init__(self, data, delta=5, num_pairs=20, threshold=None,
                pair_setting=1, sample_list=0, mixture=False):
        #self.molgraphs = 
        to_molgraph(data)
        # sometimes we use train data loader during testing, then no need to recompute threshold
        # provided threshold is used in this setting
        self.data = data
        self.threshold = get_thresholds(data, delta) if not threshold else threshold
        set_labels(data, self.threshold)
        self.cell_ids =list(self.threshold.keys())
        self.num_pairs = num_pairs
        self.pair_setting = pair_setting
        self.sample_list = sample_list
        self.mixture = mixture
        self.sens, self.insens = defaultdict(list), defaultdict(list)

        for d in data:
            self.sens[d.clid].append(d) if d.label == 1 else self.insens[d.clid].append(d)

    def __len__(self):
        return len(self.cell_ids)
        #return sum([len(_) for _ in self.sens.values()]) + sum([len(_) for _ in self.insens.values()])

    def __getitem__(self, idx):
        # get one cell line based on the passed index from sampler of loader
        clid = self.cell_ids[idx]
        if self.pair_setting == -1:
            # no pairing in ListNet setting
            if self.sample_list:
                pairs = self.sens[clid] + random.sample(self.insens[clid], self.sample_list)
            else:
                pairs = self.sens[clid]+self.insens[clid]
        elif self.pair_setting == 0:
            pairs = create_pairs_random(self.sens[clid]+self.insens[clid], self.num_pairs)
        else:
            pairs = create_pairs(self.sens[clid], self.insens[clid], self.num_pairs, self.mixture)

            if self.pair_setting == 2:
                # order does not matter among pairs from same class
                pairs.extend(create_pairs_sameg(self.sens[clid], self.num_pairs))
                pairs.extend(create_pairs_sameg(self.insens[clid], num_pairs=1))

            elif self.pair_setting == 3:
                # only create ordered pairs from same class of compounds
                pairs.extend(create_pairs_sens_ordered(self.sens[clid], self.num_pairs))
                pairs.extend(create_pairs_sens_ordered(self.insens[clid], num_pairs=1))
        np.random.shuffle(pairs)
        return pairs

    def normalize_features(self, scaler=None, scale_fatom=False, scale_fbond=False):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch_data):
        return list(chain(*batch_data))


class MoleculeDatasetTest(Dataset):
    def __init__(self, data, delta=5, threshold=None):
        to_molgraph(data)
        self.data = np.array(data)
        # for test dataset auc threshold per cell line must be based on the training dataset for setup1
        # for setup 2, threshold will be computed from test data
        self.threshold = threshold if threshold else get_thresholds(data, delta)

        set_labels(self.data, self.threshold) # set labels since get_pairs() is not called

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize_features(self, scaler, scale_fatom=False, scale_fbond=False):
        raise NotImplementedError

    @staticmethod
    def collate_fn(data):
        return data


def to_batchgraph(smiles):
    return BatchMolGraph([SMILES_TO_GRAPH[sm] for sm in smiles])
