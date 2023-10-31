import pdb
import numpy as np
from sklearn.linear_model import Lasso, ElasticNet
from argparse import ArgumentParser

parser = ArgumentParser('Elastic Net for gene selection.')
parser.add_argument('--data_dir', help='Path to the data directory.')
parser.add_argument('--l1_ratio', type=float)
parser.add_argument('--genexp_file', help='Path to the gene expression data.',
                    default='/fs/ess/PCON0041/Vishal/DrugRank/data/CCLE/CCLE_expression.csv')
#parser.add_argument('--out', help='Output file to store list of selected genes.')
#parser.set_defaults(data_dir='data/prism/LRO/')
args=parser.parse_args()

auc_matrix = np.genfromtxt(args.data_dir+'auc_matrix.txt', delimiter=',', dtype=float)
cells = np.genfromtxt(args.data_dir+'cells.txt', dtype=str)
drugs = np.genfromtxt(args.data_dir+'drugs.txt', dtype=str)

expression = {}
genes = []
sep = ',' if 'CCLE' in args.genexp_file else '\t'
with open(args.genexp_file, 'r') as fp:
    tmp = np.array(next(fp).strip().split(sep)[1:])
    genes = np.array([_.split()[0] for _ in tmp])

    for line in fp.readlines():
        tmp = line.strip().split(sep)
        expression[tmp[0]] = np.array(tmp[1:], float)

coeff = []

for i, drug in enumerate(drugs):
    sens = auc_matrix[:,i]
    ind = np.argwhere(~np.isnan(sens)).flatten()
    cell_ids = cells[ind]

    X, y = [], []
    for j, cid in zip(ind, cell_ids):
        X.append(expression[cid])
        y.append(sens[j])

    #print(f'Fitting for drug: {drug}')
    enet = ElasticNet(random_state=0, max_iter=1000, l1_ratio=0.1)
    enet.fit(X, y)
    coeff.append(enet.coef_)

coeff = np.asarray(coeff)
print('Done fitting')

# select genes with non-zero coefficient
sel_genes = genes[np.sum(coeff, axis=0).nonzero()[0]]
np.savetxt(args.data_dir+'genes.txt', sel_genes, fmt='%s', delimiter='\n')