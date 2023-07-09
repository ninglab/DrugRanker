import numpy as np
import torch
import torch.nn as nn

from dataloader.loader import to_batchgraph
from features.features_generators import *
from features.featurization import mol2graph
from utils.nn_utils import compute_pnorm, compute_gnorm
from utils.common import pair2set

import time

def train_step_listnet(clobj, model, loader, criterion, optimizer, args):
    """
    train listwise models
    `clobj`: cell line object
    `model`: model object
    `loader`: data loader
    `criterion`: loss function
    `optimizer`: optimizer  
    """
    model.train()
    total_loss = 0
    iters, gnorm = 0, 0

    model.zero_grad()
    for i, batch in enumerate(loader):
        batch_loss = 0
        clids, mols, features, labels, aucs = [], [], [], [], []
        for d in batch:
            aucs.append(d.auc)
            mols.append(d.smiles)
            features.append(d.features)
            clids.append(d.clid)
            labels.append(d.label)

        cl_emb = torch.from_numpy(np.array(clobj.get_expression(clids))).to(args.device)

        # batch graph needed only for gnn models
        molgraph = to_batchgraph(mols) if args.gnn else None

        pred = model(cl_emb, cmp1=molgraph, smiles1=mols, feat1=features, output_type=0)
        if args.model == 'listone':
            batch_loss = criterion(pred, torch.tensor(aucs, device=pred.device))
        elif args.model == 'listall':
            batch_loss = criterion(pred.reshape(1,-1), torch.tensor(labels, device=pred.device).reshape(1,-1))
        else:
            raise ValueError('Invalid listwise model name')
        total_loss += batch_loss.item()

        batch_loss.backward()
        if (iters+1) == len(loader) or (iters+1)%args.gradient_steps==0:
            optimizer.step()
            gnorm = compute_gnorm(model)
            model.zero_grad()
        iters += 1

    return total_loss/iters, gnorm


def train_step(clobj, model, loader, criterion, optimizer, args):
    """
    train pairwise models
    `clobj`: cell line object
    `model`: model object
    `loader`: data loader
    `criterion`: loss function
    `optimizer`: optimizer
    """
    model.train()
    total_loss = 0
    iters, gnorm = 0, 0
    labels = None
    bce = nn.BCEWithLogitsLoss()

    for i, batch in enumerate(loader):
        ccl_ids = []
        mols1, mols2, features1, features2, labels1, labels2, aucs1, aucs2 = [], [], [], [], [], [], [], []
        for d1, d2 in batch:
            ccl_ids.append(d1.clid)
            mols1.append(d1.smiles)
            features1.append(d1.features)
            labels1.append(d1.label)
            aucs1.append(d1.auc)

            mols2.append(d2.smiles)
            features2.append(d2.features)
            labels2.append(d2.label)
            aucs2.append(d2.auc)

        cl_emb = torch.from_numpy(np.asarray(clobj.get_expression(ccl_ids))).to(args.device)
        # sign = 1 if 1st comp is more sensitive than the 2nd comp; else -1
        sign = torch.from_numpy(np.sign(np.array(aucs2) - np.array(aucs1))).to(args.device)
        # y = 1 if both the comp in a pair are of same label, else 0
        y = torch.from_numpy(np.array(np.array(labels1) == np.array(labels2), dtype=int)).to(args.device)

        if args.model == 'pairpushc':
            # to reduce call to batch and self.gnn, convert pairs to sets of graphs and features
            pos, neg, list_mols, list_features = pair2set(batch)
            molgraph = to_batchgraph(list_mols) if args.gnn else None 

            pred_diff, plabel, clabel, cmp_sim = model(cl_emb, cmp1=molgraph,
                                                    smiles1=list_mols, feat1=list_features,
                                                        pos=pos, neg=neg)
        else:
            raise ValueError('Invalid model name')

        batch_loss = 0

        #actual_diff = torch.from_numpy(np.array(sens1) - np.array(sens2)).to(args.device)
        if args.surrogate == 'logistic':
            # sign should be 1 if (+,-) pair and -1 if (-,+) pair
            batch_loss = torch.mean(criterion(-sign*pred_diff), dim=0)
        elif args.surrogate == 'tcbb':
            batch_loss = criterion(pred_diff, labels1, labels2, sign)
        else:
            raise ValueError('Invalid surrogate loss name')

        #elif args.surrogate == 'margin':
        #    batch_loss += criterion(pred_diff, y, sign)

        # drug pair classification loss: not used in paper
        if args.classify_pairs:
            bce_loss = bce(plabel, y.float())
            batch_loss += bce_loss

        # drug instance sensitivity classification loss
        if args.classify_cmp:
            clabel = clabel.flatten()
            labels = torch.Tensor(np.array(labels1 + labels2)).to(clabel.device)
            batch_loss += bce(clabel, labels.float())

        ## regularization
        if args.regularization:
            batch_loss = batch_loss + args.regularization*compute_pnorm(model)**2

        batch_loss.backward()
        if (iters+1) == len(loader) or (iters+1)%args.gradient_steps==0:
            optimizer.step()
            gnorm = compute_gnorm(model)
            model.zero_grad()

        total_loss += batch_loss.item()
        iters += 1

    return total_loss/iters, gnorm
