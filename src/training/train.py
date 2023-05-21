import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from dataloader.loader import to_batchgraph
from features.features_generators import *
from features.featurization import mol2graph
from utils.nn_utils import compute_pnorm, compute_gnorm
from utils.common import pair2set

import time

def train_step_listnet(clobj, model, loader, criterion, optimizer, args):
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

        molgraph = to_batchgraph(mols) if args.gnn else None

        pred = model(cl_emb, cmp1=molgraph, smiles1=mols, feat1=features, output_type=0)
        if args.model == 'listone':
            batch_loss = criterion(pred, torch.tensor(aucs, device=pred.device))
        elif args.model == 'listall':
            batch_loss = criterion(pred.reshape(1,-1), torch.tensor(labels, device=pred.device).reshape(1,-1))
        total_loss += batch_loss.item()

        batch_loss.backward()
        if (iters+1) == len(loader) or (iters+1)%args.gradient_steps==0:
            optimizer.step()
            gnorm = compute_gnorm(model)
            #print(gnorm)
            model.zero_grad()
        iters += 1

    return total_loss/iters, gnorm

def train_step(clobj, model, loader, criterion, optimizer, args):
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

        if args.model == 'xattnet-n':  # node level update
            pos, neg = None, None
            ## create unique pairs to reduce overload
            pairs2set = set()
            for a,b in zip(mols1, mols2):
                if (a,b) not in pairs2set:
                    pairs2set.add((a,b))
            pairs2set = list(pairs2set)
            mols1 = [_[0] for _ in pairs2set]
            mols2 = [_[1] for _ in pairs2set]
            pos, neg = [], []
            for d in batch:
                pos.append(mols1.index(d[0].smiles))
                neg.append(mols2.index(d[1].smiles))

            molgraph1, molgraph2 = None, None
            if (not args.use_features_only) and args.gnn:
                #start = time.time()
                molgraph1, molgraph2 = to_batchgraph(mols1), to_batchgraph(mols2)
                #print(f'to batchgraph: Batch: {i} Time: {time.time()-start}')
            #start = time.time()
            pred_diff, plabel, clabel, cmp_sim = model(cl_emb, cmp1=molgraph1, smiles1=mols1, feat1=features1, \
                                                cmp2=molgraph2, smiles2=mols2, feat2=features2, pos=pos, neg=neg)
            #print(f'Forward pass: Batch: {i} Time: {time.time()-start}')


        elif args.model == 'ranknet' or args.model == 'xattnet-g':
            pos, neg, list_mols, list_features = pair2set(batch)
            molgraph = None
            if not args.use_features_only and args.gnn:
                #start = time.time()
                molgraph = to_batchgraph(list_mols)
                #print(f'to batchgraph: Batch: {i} Time: {time.time()-start}')
            #start = time.time()
            pred_diff, plabel, clabel, cmp_sim = model(cl_emb, cmp1=molgraph, smiles1=list_mols, feat1=list_features, \
                                                pos=pos, neg=neg) # reduce call to self.gnn
            #print(f'Forward pass: Batch: {i} Time: {time.time()-start}')
            """
            if not args.use_features_only and args.gnn == 'dmpn':
                molgraph1, molgraph2 = to_batchgraph(mols1), to_batchgraph(mols2)
            pred_diff, plabel, clabel, _ = model(cl_emb, cmp1=molgraph1, smiles1=mols1, feat1=features1, \
                            cmp2=molgraph2, smiles2=mols2, feat2=features2)
            """


        batch_loss = 0

        #actual_diff = torch.from_numpy(np.array(sens1) - np.array(sens2)).to(args.device)
        if args.surrogate == 'logistic':
            # sign should be 1 if (+,-) pair and -1 if (-,+) pair
            batch_loss = torch.mean(criterion(-sign*pred_diff), dim=0)

        elif args.surrogate == 'tcbb' or args.surrogate == 'tcbb1':
            batch_loss = criterion(pred_diff, labels1, labels2, sign)

        elif args.surrogate == 'margin':
            batch_loss += criterion(pred_diff, y, sign)

        if args.cluster:
            cr = HingeLoss(args.margin)
            #hinge_loss = torch.mean(cr(torch.abs(pred_diff), y))
            hinge_loss = torch.mean(cr(cmp_sim, y))
            batch_loss += args.gamma*hinge_loss

        if args.classify_pairs:
            bce_loss = bce(plabel, y.float())
            batch_loss += bce_loss

        if args.classify_cmp:
            clabel = clabel.flatten()
            labels = torch.Tensor(np.array(labels1 + labels2)).to(clabel.device)
            batch_loss += bce(clabel, labels.float())

        ## regularization
        if args.regularization:
            batch_loss = batch_loss + args.regularization*compute_pnorm(model)**2

        #start = time.time()
        batch_loss.backward()
        #model.zero_grad()
        #print(f'Backward pass: Batch: {i} Time: {time.time()-start}')
        '''
        if args.grad_clip:
            clip_grad_norm_([p for p in model.parameters() if p.grad is not None], args.grad_clip)
        '''

        if (iters+1) == len(loader) or (iters+1)%args.gradient_steps==0:
            optimizer.step()
            gnorm = compute_gnorm(model)
            model.zero_grad()

        total_loss += batch_loss.item()
        iters += 1

    return total_loss/iters, gnorm
