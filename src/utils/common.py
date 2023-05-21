import sys
import os
import logging
import json

from collections import defaultdict
from argparse import Namespace

import torch
import numpy as np
from sklearn.metrics import ndcg_score
from features.features_generators import *

from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from scipy import stats
import seaborn as sns
import colorcet as cc

# for reproducibility
def set_seed(seed=123):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    import random
    random.seed(seed)

def set_logger(args=None):
    '''
    Write logs to checkpoint and console
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format   = '%(asctime)s %(levelname)-8s %(message)s',
        level    = logging.INFO,
        datefmt  = '%Y-%m-%d %H:%M:%S',
        filename = log_file,
        filemode = 'w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def save_model(model, optimizer, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    if args.device == torch.device('cpu'):
        argparse_dict['device'] = 'cpu'
    else:
        argparse_dict['device'] = 'cuda:0'

    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'model.pt')
    )


def load_model(model, path, device):
    logging.info('Loading checkpoint %s...' % path)
    checkpoint = torch.load(os.path.join(path, 'model.pt'), map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)


def create_pairs(dataloader):
    x_train = []
    for data1 in dataloader:
        for data2 in dataloader:
            if data1[2] < data2[2]:
                x_train.append(list(zip(*[data1, data2])))
    if x_train:
        return np.array(x_train)
    else:
        return None

def get_pair_setting(args):
    # pair setting (`ps`) determines what kind of pairs to fit and evaluate the model
    train_ps, test_ps = 0, 0

    if args.surrogate == 'logistic':
        train_ps = 1
    elif args.surrogate == 'tcbb':
        train_ps = 2
    if args.model in ['listone', 'listall']:
        train_ps = -1
    return train_ps, test_ps


def calc_avg_perf(metrics):
    avg_metric = defaultdict(list)
    for metric in metrics:
        for k, v in metric.items():
            if k in avg_metric:
                avg_metric[k].append(v)
            else:
                avg_metric[k] = [v]

    return {k: np.mean(v) for k, v in avg_metric.items()}


def write_to_file(pred_scores_list, true_scores_list, args, name, fold):
    logfile = os.path.join(args.save_path, name)

    assert(len(pred_scores_list) == len(true_scores_list))
    mode = 'w'
    if fold > 1:
        mode = 'a'

    with open(logfile, mode) as fp:
        for i in range(len(pred_scores_list)):
            print(' '.join(map(str, pred_scores_list[i])), file=fp)
            print(' '.join(map(str, true_scores_list[i])), file=fp)


def pair2set(batch):

    mols1 = [d.smiles for d in list(zip(*batch))[0]]
    mols2 = [d.smiles for d in list(zip(*batch))[1]]

    features1 = [d.features for d in list(zip(*batch))[0]]
    features2 = [d.features for d in list(zip(*batch))[1]]

    list_mols = list(set(mols1+mols2))

    pos, neg = [], []

    for mol1, mol2 in zip(mols1, mols2):
        pos.append(list_mols.index(mol1))
        neg.append(list_mols.index(mol2))

    list_features = [None]*len(list_mols)
    for i, (f1, f2) in enumerate(list(zip(features1, features2))):
        if f1 is not None:
            list_features[pos[i]] = f1[:]
        if f2 is not None:
            list_features[neg[i]] = f2[:]

    return pos, neg, list_mols, list_features


def precompute_features(args):
    if args.feature_gen is None:
        smiles_f = {}
        with open(args.smiles_path, 'r') as fp:
            next(fp)
            for line in fp.readlines():
                smiles = line.strip().split('\t')[-1]
                smiles_f[smiles] = None
        return smiles_f


    features_generator = get_features_generator(args.feature_gen)

    smiles_f = {}
    with open(args.smiles_path, 'r') as fp:
        next(fp)
        for line in fp.readlines():
            smiles = line.strip().split('\t')[-1]

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and mol.GetNumHeavyAtoms() > 0:
                smiles_f[smiles] = features_generator(mol)

    return smiles_f


def tsne(cell_emb, path, types=None):
    tsne = TSNE(n_components=2, init='pca', random_state=0, n_iter=5000, perplexity=10, learning_rate='auto')
    cell_emb = tsne.fit_transform(cell_emb)
    plt.figure(figsize=(6,5))
    if types is not None:
        sns.scatterplot(x=cell_emb[:, 0], y=cell_emb[:, 1], hue=types, 
                        palette=sns.color_palette(cc.glasbey, n_colors=len(set(types))))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') #, labels=t, handles=scatter.legend_elements()[0])
    else:
        sns.scatterplot(x=cell_emb[:, 0], y=cell_emb[:, 1])
    plt.savefig(path, bbox_inches='tight')


def elbow_kmeans(X, data, K, transform, save_dir, p):
    inertia = {}
    for k in K:
        km = KMeans(n_clusters=k, init='random', random_state=0, max_iter=2000).fit(X)
        inertia[k] = km.inertia_
        print(f"Inertia with k = {k}: {km.inertia_:.4f}")

        assignments = km.labels_
        classes = np.unique(assignments)
        plt.figure()
        sns.scatterplot(x=data[:,0], y=data[:,1], hue=assignments,
                        palette=sns.color_palette(cc.glasbey, len(classes)))
        plt.title(f"K-means clustering on {transform}-reduced data with k = {k}")
        plt.legend('')
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') #, labels=t, handles=scatter.legend_elements()[0])
        if transform == 'tsne':
            plt.savefig(save_dir + f'/elbow-{transform}-{p}/cluster_{k}.jpg', bbox_inches='tight')
        else:
            plt.savefig(save_dir + f'/elbow-{transform}/cluster_{k}.jpg', bbox_inches='tight')

    kbest = min(inertia, key=inertia.get)
    print(f"lowest inertia with k = {kbest}")
    return kbest

def cluster(X, save_dir, plot_type, labels=[],
            algo='kmeans', transform='pca', elbow=False, p=10):
    #from sklearn.preprocessing import LabelEncoder
    #le = LabelEncoder()
    #le.fit(list(set(labels)))
    if (transform == 'tsne') and (not os.path.exists(save_dir+f'/elbow-{transform}-{p}')):
        os.makedirs(save_dir+f'/elbow-{transform}-{p}')

    k = 25 if len(labels)==0 else len(np.unique(labels))

    pca = PCA(n_components=2, random_state=0)
    tsne = TSNE(n_components=2, init='pca', random_state=0, learning_rate='auto', perplexity=p, n_iter=5000)

    if transform == 'pca':
        data = pca.fit_transform(X).astype(np.float64)
    elif transform == 'tsne':
        data = tsne.fit_transform(X)

    if elbow:
        elbow_kmeans(X, data, [2,3,4,5,10,15,20,25,30,50,100,200], transform, save_dir, p)

    if algo == 'kmeans':
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0, max_iter=2000).fit(X)
        assignments = kmeans.labels_
        centroids = kmeans.cluster_centers_
    elif algo == 'dbscan':
        dbscan = DBSCAN(eps=0.3).fit(X)
        assignments = dbscan.labels_
        centroids = dbscan.core_sample_indices_

    #classes = np.unique(assignments)
    classes = np.unique(labels)
    plt.figure()
    sns.scatterplot(x=data[:,0], y=data[:,1], hue=labels, #hue=assignments,
                    palette=sns.color_palette(cc.glasbey, len(classes)))

    plt.title(f"K-means clustering on {transform}-reduced data")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') #, labels=t, handles=scatter.legend_elements()[0])
    if transform == 'tsne':
        plt.savefig(save_dir + f'/cluster_{plot_type}_{transform}-{p}.jpg', bbox_inches='tight')
    else:
        plt.savefig(save_dir + f'/cluster_{plot_type}_{transform}.jpg', bbox_inches='tight')
    np.savetxt(save_dir + f'/assignments_{plot_type}.txt', assignments, delimiter='\n', fmt='%d')

    if len(labels):
        import pandas as pd
        confusion_matrix = pd.crosstab(labels, assignments, normalize='columns').sort_index()
        counts0 = pd.crosstab(labels, assignments).sort_index().sum(axis=0)
        counts1 = pd.crosstab(labels, assignments).sort_index().sum(axis=1)
        ylabels = list(counts1.items())
        xlabels = list(counts0.items())

        entr = stats.entropy(confusion_matrix)
        # most impure clusters
        print(entr)
        print(np.argsort(-entr)[:5])

        # plot overlap of ground truth labels with predicted clusters
        plt.figure(figsize=(20,25))
        sns.heatmap(confusion_matrix, square=True, annot=True, fmt='.2f',
                    linewidths=0.1, yticklabels=ylabels, xticklabels=xlabels)
        plt.savefig(save_dir+f'/overlap_{plot_type}.jpg', bbox_inches='tight')

        # plot similarity of dist over clusters for each group
        from sklearn.metrics.pairwise import cosine_similarity
        #confusion_matrix = pd.crosstab(labels, assignments, normalize='index').sort_index()
        simc = cosine_similarity(confusion_matrix)
        from scipy.spatial.distance import jensenshannon
        #simc = np.zeros(confusion_matrix.shape)
        #for i in range(confusion_matrix.shape[0]):
        #    for j in range(confusion_matrix.shape[1]):
        #        simc[i][j] = 1-jensenshannon(confusion_matrix[i], confusion_matrix[j], 2)
        #simc = [np.corrcoef(A[i], B[i])[0][1] for i in range(A.shape[0])]

        plt.figure(figsize=(20,25))
        sns.heatmap(simc, square=True,
                    linewidths=0.1, yticklabels=ylabels, xticklabels=ylabels)
        plt.savefig(save_dir+f'/cm_{plot_type}.jpg', bbox_inches='tight')
        np.savetxt(save_dir+f'/cm_{plot_type}.txt', simc, delimiter=',', fmt='%.3f')
