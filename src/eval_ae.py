import numpy as np
import torch
import torch.nn as nn
import json
from models.ae import AE
from models.ranknet import RankNet
from utils.common import load_model, tsne, cluster
from dataloader.loader import CellLine
from argparse import ArgumentParser, Namespace
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import seaborn as sns

def eval_ae(model, data):
    model.eval()
    with torch.no_grad():
        emb = model(data, use_encoder_only=True)
    return emb

def compute_sim(exp, sim='rbf', sigma=10):
    sim_matrix = np.zeros((exp.shape[0], exp.shape[0]))
    if sim == 'cos':
        sim_matrix = cosine_similarity(exp)
    elif sim == 'dot':
        sim_matrix = exp.dot(exp.T)
    elif sim == 'rbf':
        gamma = 1/sigma**2
        pdist = pairwise_distances(exp, metric='l2')
        sim_matrix = np.exp(-gamma*pdist)
    return sim_matrix

parser = ArgumentParser()
parser.add_argument('--config_path', required=True)
parser.add_argument('--ckpt_path', required=True)
parser.add_argument('--baseline_emb', required=True)
parser.add_argument('--cells', required=True)
parser.add_argument('--sel_genes', required=True)
parser.add_argument('--ana_dir', required=True)

args = parser.parse_args()

with open(args.config_path) as f:
    config = json.loads(f.read())

config.update(vars(args))
args = Namespace(**config)
args.device = 'cpu'

# load trained ranking model
model = RankNet(args).to(args.device)
model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device('cpu')))
ae = model.ae

# load pretrained AE
pretrained_ae = AE(args)
pretrained_ae.to(args.device)
load_model(pretrained_ae, args.trained_ae_path, args.device)

# load gene expression data
data = np.genfromtxt('/fs/ess/PCON0041/Vishal/DrugRank/data/CCLE/CCLE_expression.csv',
                         delimiter=',', dtype=str)
# get cell ID and type=tissue/cancer mappings
# The cell IDs order must be consistent with that in  U.txt from CCLERank
cell_types = np.genfromtxt(args.cells, delimiter='\t', dtype=str)
cell_ids = cell_types#[:,0]
#types = cell_types[:,1]

# only use the required cells
#idx = np.in1d(data[:,0], cell_ids)  # WRONG!!! this does not order `data` acc to `cell_types`
idx = (data[:,0] == cell_ids[:,None]).nonzero()[1]
gene_exp = np.array(data[idx, 1:], float)
all_genes = np.asarray([_.split()[0] for _ in data[0][1:]])
exp = torch.Tensor(gene_exp, device=args.device)

emb_pt = eval_ae(pretrained_ae, exp).cpu().numpy()
emb_tu = eval_ae(ae, exp).cpu().numpy()

np.savetxt(args.ana_dir+'pretrained_U.txt', emb_pt, delimiter=',', fmt='%.4f')
np.savetxt(args.ana_dir+'tuned_U.txt', emb_tu, delimiter=',', fmt='%.4f')

with torch.no_grad():
    emb_tr = model.scoring.ffn(eval_ae(ae, exp)).cpu().numpy()

np.savetxt(args.ana_dir+'tuned_tr_U.txt', emb_tr, delimiter=',', fmt='%.4f')

emb_bs = np.genfromtxt(args.baseline_emb, delimiter=',', dtype=float)

# only select the cancer genes to compute GE similarity
genes = np.genfromtxt(args.sel_genes, delimiter='\n', dtype=str)
gene_idx = np.in1d(all_genes, genes)
gene_exp = gene_exp[:, gene_idx]


# compute similarities using GE, LV-pretained, LV-trained and LV-baseline
sim_ge = compute_sim(gene_exp)
sim_vp = compute_sim(emb_pt)
sim_vt = compute_sim(emb_tu)
sim_vtr = compute_sim(emb_tr)
sim_vb = compute_sim(emb_bs)
np.savetxt(args.ana_dir+'csim_ge.txt', sim_ge, delimiter=',', fmt='%.4f')
np.savetxt(args.ana_dir+'csim_vp.txt', sim_vp, delimiter=',', fmt='%.4f')
np.savetxt(args.ana_dir+'csim_vt.txt', sim_vt, delimiter=',', fmt='%.4f')
np.savetxt(args.ana_dir+'csim_vtr.txt', sim_vtr, delimiter=',', fmt='%.4f')
np.savetxt(args.ana_dir+'csim_vb.txt', sim_vb, delimiter=',', fmt='%.4f')
"""
cluster(emb_pt, args.ana_dir, 'pretrained', types, transform='pca')
cluster(emb_tr, args.ana_dir, 'tuned', types, transform='pca', elbow=True)

for p in [5,10,20,30,50,100]:
    cluster(emb_pt, args.ana_dir, 'pretrained', types, transform='tsne', p=p)
    cluster(emb_tr, args.ana_dir, 'tuned', types, transform='tsne', elbow=True, p=p)
#cluster(emb_bs, args.ana_dir, 'cclerank', types, transform='tsne')

"""
## run tsne
#tsne(emb_pt, args.ana_dir+'tsne_pretrained.jpg', types)
#tsne(emb_tr, args.ana_dir+'tsne_tuned.jpg', types)
#tsne(emb_bs, args.ana_dir+'tsne_cclerank.jpg', types)
