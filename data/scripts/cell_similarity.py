import os,sys
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from argparse import ArgumentParser
from scipy.stats import spearmanr

parser = ArgumentParser('Computing pairwise cell line similarity.')
parser.add_argument('--data_dir', help='Path to the data directory containing AUC matrix, cell IDs')
parser.add_argument('--genexp_file', help='Path to the gene expression file',
                    default='/fs/ess/PCON0041/Vishal/DrugRank/data/CCLE/CCLE_expression.csv')
parser.add_argument('--genes', help='Path to the file containing list of genes')
parser.add_argument('--sim', choices=['cos', 'rbf', 'ds'], help='Similarity method')
parser.add_argument('--sigma', default=10, help='Sigma for RBF')

args = parser.parse_args()

genes = np.genfromtxt(args.genes, delimiter='\n', dtype=str)
cells = np.genfromtxt(args.data_dir+'cells.txt', delimiter='\n', dtype=str)
matrix = np.genfromtxt(args.data_dir+'auc_matrix.txt', delimiter=',', missing_values='nan')

gen_exp = {}
sep = ',' if 'CCLE' in args.genexp_file else '\t'
with open(args.genexp_file, 'r') as fp:
    tmp = np.array(next(fp).strip().split(sep)[1:])
    all_genes = np.array([_.split()[0] for _ in tmp])

    for line in fp:
        tmp = line.strip().split(sep)
        gen_exp[tmp[0]] = np.array(tmp[1:], dtype=float)

# get the correct indices of cancer genes in the order they appear in `all_genes`
gene_idx = (all_genes == genes[:,None]).nonzero()[1]
#gene_idx = np.in1d(all_genes, genes).nonzero()[0]
exp = np.zeros((len(cells), len(genes)))  # M cells x G genes

#print(genes[:5], all_genes[gene_idx[:5]])
# order the cells in gene exp matrix acc to the order in `cells.txt`
for i, cell in enumerate(cells):
    exp[i] = gen_exp[cell][gene_idx].copy()

#compute similarity matrix
out = f'sim_{args.sim}'
sim_matrix = np.zeros((exp.shape[0], exp.shape[0]))
if args.sim == 'cos':
    sim_matrix = cosine_similarity(exp)
    out += '.txt'
elif args.sim == 'rbf':
    gamma = 1/args.sigma**2
    pdist = pairwise_distances(exp, metric='l2')
    sim_matrix = np.exp(-gamma*pdist)
    out += f'_{args.sigma}.txt'
elif args.sim == 'ds':
    #matrix = np.nan_to_num(matrix, nan=-1)
    sim_matrix = spearmanr(matrix.T, nan_policy='omit')[0] #for j in range(matrix.shape[0])] \
                    #for i in range(matrix.shape[0])]
    out += '.txt'

np.savetxt(args.data_dir+out, sim_matrix, delimiter=',', fmt='%.6f')
