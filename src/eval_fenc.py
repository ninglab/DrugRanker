import json
import torch
import numpy as np
from argparse import ArgumentParser, Namespace
from models.ranknet import RankNet
from utils.common import precompute_features, tsne, cluster
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from rdkit.Chem import DataStructs, AllChem, MolFromSmiles

def compute_sim(emb, sim='rbf', sigma=1):
    sim_matrix = np.zeros((emb.shape[0], emb.shape[0]))
    if sim == 'cos':
        sim_matrix = cosine_similarity(emb)
    elif sim == 'rbf':
        gamma = 1/sigma**2
        pdist = pairwise_distances(emb, metric='l2')
        sim_matrix = np.exp(-gamma*pdist)
    return sim_matrix

def compute_tanimoto(smiles):
    fps = [AllChem.GetHashedMorganFingerprint(MolFromSmiles(_), radius=3, nBits=2048)
                for _ in smiles]
    tanimoto = []
    for i,f in enumerate(fps):
        tanimoto.append(DataStructs.BulkTanimotoSimilarity(f, fps))
    return tanimoto


parser = ArgumentParser()
parser.add_argument('--ckpt_path', required=True)
parser.add_argument('--config_path', required=True)
parser.add_argument('--baseline', required=True)
parser.add_argument('--smiles_path', required=True)
parser.add_argument('--cmpd', required=True)
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
fenc = model.enc

features = precompute_features(args)
# load cmp SMILES data
cmpds = np.genfromtxt(args.smiles_path, delimiter='\t', comments=None, dtype=str)
cmp_ids = cmpds[:,0]
smiles = cmpds[:,-1]

# Use the ordering of drugs as in the drugs.txt/V.txt
drugs = np.genfromtxt(args.cmpd, delimiter='\n', dtype=str)
#idx = np.in1d(cmp_ids, drugs) WRONG this does not give indices in order of `drugs`
idx = (cmp_ids == drugs[:,None]).nonzero()[1]
smiles = smiles[idx]
fg = np.asarray([features[_] for _ in smiles])
inputs = torch.Tensor(fg, device=args.device)
with torch.no_grad():
    emb = fenc(smiles, inputs).cpu().numpy()

np.savetxt(args.ana_dir+'V.txt', emb, delimiter=',', fmt='%.4f')
emb_bs = np.genfromtxt(args.baseline+'/V.txt', delimiter=',', dtype=float).T
# compute similarities using GE, LV-pretained, LV-trained and LV-baseline
sim_fp = compute_tanimoto(smiles)
sim_vt = compute_sim(emb)
sim_vb = compute_sim(emb_bs)
np.savetxt(args.ana_dir+'dsim_fp.txt', sim_fp, delimiter=',', fmt='%.4f')
np.savetxt(args.ana_dir+'dsim_vt.txt', sim_vt, delimiter=',', fmt='%.4f')
np.savetxt(args.ana_dir+'dsim_vb.txt', sim_vb, delimiter=',', fmt='%.4f')


# run tsne
#tsne(emb, args.ana_dir+'trained.jpg')
#tsne(emb_bs, args.ana_dir+'cclerank.jpg')
#cluster(emb, args.ana_dir, 'tuned', algo='kmeans', elbow=True, transform='tsne')
#cluster(emb_bs, args.ana_dir, 'cclerank', algo='kmeans', elbow=True)
