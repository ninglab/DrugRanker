import os
import sys
import numpy as np
from argparse import ArgumentParser

## run this as
## python create_setup_LCO.py --data_dir ../data/prism/LRO/pletorg/

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, help='path to data directory')
parser.add_argument('--save_dir', type=str, help='path to save directory')
parser.set_defaults(data_dir='data/prism/LRO/', save_dir='data/prism/LCO/pletorg/')
args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir

cancers = np.genfromtxt(data_dir+'cancer.txt', delimiter='\n', dtype=str)
cells = np.genfromtxt(data_dir+'cells.txt', delimiter='\n', dtype=str)
drugs = np.genfromtxt(data_dir+'drugs.txt', delimiter='\n', dtype=str)
auc = np.genfromtxt(data_dir+'auc_matrix.txt', delimiter=',', dtype=np.float32)
csim_cos = np.genfromtxt(data_dir+'sim_cos.txt', delimiter=',')
csim_rbf = np.genfromtxt(data_dir+'sim_rbf_10.txt', delimiter=',')

SEED = 123
np.random.seed(SEED)
FOLDS = 5
set_cancer = set(cancers)

# to hold lists of indices for each fold
cv_train_idx, cv_val_idx, cv_test_idx = [[] for _ in range(FOLDS)], \
            [[] for _ in range(FOLDS)], [[] for _ in range(FOLDS)]

for canc in set_cancer:
    idx = (canc == cancers).nonzero()[0]
    # less than 5 cell lines in a cancer type, ignore that type
    if len(idx) < FOLDS:
        print(f'Cancer type {canc} excluded')
        continue
    np.random.shuffle(idx)
    N = len(idx)
    splits = np.array(np.array_split(idx, FOLDS))

    for fold in range(FOLDS):
        # first N//FOLDS of idx will be test, next N//FOLDS will be val, rest will be train
        cv_test_idx[fold] += splits[fold].tolist()
        cv_val_idx[fold] += splits[(fold+1)%FOLDS].tolist()
        mask = np.ones(FOLDS, dtype=bool)
        mask[fold] = mask[(fold+1)%FOLDS] = False
        cv_train_idx[fold] += np.concatenate(splits[mask]).tolist()
        
cv_train_idx = np.asarray(cv_train_idx)
cv_test_idx = np.asarray(cv_test_idx)
cv_val_idx = np.asarray(cv_val_idx)

for fold in range(FOLDS):
    if not os.path.exists(f'{save_dir}/fold_{fold}/'):
        os.makedirs(f'{save_dir}/fold_{fold}/')

    train_idx, val_idx, test_idx = cv_train_idx[fold], cv_val_idx[fold], cv_test_idx[fold]
    # save cell lines
    np.savetxt(f'{save_dir}/fold_{fold}/train_cells.txt', cells[train_idx], delimiter='\n', fmt='%s')
    np.savetxt(f'{save_dir}/fold_{fold}/val_cells.txt', cells[val_idx], delimiter='\n', fmt='%s')
    np.savetxt(f'{save_dir}/fold_{fold}/test_cells.txt', cells[test_idx], delimiter='\n', fmt='%s')

    # create the similarity matrix for this setup only for training cell lines
    np.savetxt(f'{save_dir}/fold_{fold}/train_sim_cos.txt', csim_cos[train_idx][:,train_idx], delimiter=',')
    np.savetxt(f'{save_dir}/fold_{fold}/train_sim_rbf_10.txt', csim_rbf[train_idx][:,train_idx], delimiter=',')

    train_auc = auc[train_idx]
    test_auc = auc[test_idx]
    val_auc = auc[val_idx]

    rel = np.nanpercentile(auc,5,axis=1,method='closest_observation')

    #'%s' = '%.15f' if (dataset == 'prism') else '%.4f'
    # create auc matrices
    np.savetxt(f'{save_dir}/fold_{fold}/train_matrix.txt', train_auc, delimiter=',', fmt='%s')
    np.savetxt(f'{save_dir}/fold_{fold}/val_matrix.txt', val_auc, delimiter=',', fmt='%s')
    np.savetxt(f'{save_dir}/fold_{fold}/test_matrix.txt', test_auc, delimiter=',', fmt='%s')

    np.savetxt(f'{save_dir}/fold_{fold}/train_rel.txt', rel[train_idx], delimiter='\n', fmt='%s')
    np.savetxt(f'{save_dir}/fold_{fold}/val_rel.txt', rel[val_idx], delimiter='\n', fmt='%s')
    np.savetxt(f'{save_dir}/fold_{fold}/test_rel.txt', rel[test_idx], delimiter='\n', fmt='%s')


# create a list of all cell lines used in this setup and respective pairwise similarity matrix for training
all_cells_idx = cv_train_idx[0] + cv_val_idx[0] + cv_test_idx[0]
np.savetxt(f'{save_dir}/../cells.txt', cells[all_cells_idx], delimiter='\n', fmt='%s')

#print(all_cells_idx)
sep = '\t' if ('prism' in data_dir or 'ctrpv2' in data_dir) else ','
auc_list = np.genfromtxt(data_dir+'aucs.txt', delimiter=sep, dtype=str)
# create the list of aucs for each cell and drug
fp = open(f'{save_dir}/../aucs.txt', 'w')
for cell, drug, s in auc_list:
    if cell in cells[all_cells_idx]:
        print(cell + sep + drug + sep + str(s), file=fp)
fp.close()