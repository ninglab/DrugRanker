#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train_ae.py
# Author            : Vishal Dey <dey.78@osu.edu>
# Date              : Tue 22 Feb 2022 21:58:09
# Last Modified Date: Tue 10 Nov 2022 21:58:09
# Last Modified By  : Vishal Dey <dey.78@osu.edu>

import sys
import os
sys.path.insert(0, ".")
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.ae import AE
from utils.common import set_seed, save_model, load_model
from utils.metrics import mse
from argparse import ArgumentParser

def read_gene_exp(file):
    """
    Read gene expression data from file
    `file` is a csv file with first column as cell ID and rest as expression values
    """
    exp = {}   # cell_id -> expression
    sep = '\t' if 'Combined' in file else ','
    with open(file, 'r') as fp:
        next(fp)
        for line in fp.readlines():
            tmp = line.strip().split(sep)
            exp[tmp[0]] = np.array(tmp[1:], dtype=np.float32) # type: ignore
    return exp


def read_splits(splits_path):
    """
    Read train and test cell IDs from file
    """
    train_ids = np.genfromtxt(splits_path+'train_cells.txt', dtype=str) # type: ignore
    test_ids = np.genfromtxt(splits_path+'val_cells.txt', dtype=str) # type: ignore
    return train_ids, test_ids


def evaluate(model, dataloader, metric, device='cpu'):
    """
    Evaluate model on given dataloader
    """
    model.eval()

    losses = []
    for batch in dataloader:
        batch = batch.to(device)
        recon = model(batch)
        l = metric(recon, batch)
        losses.append(l.item())

    return np.mean(losses)


def test_ae(args):
    """
    Test the trained model on test set of expression data
    """
    set_seed()
    data = list(read_gene_exp(args.genexp_file).values())

    model = AE(args)
    load_model(model, args.save_path, device=args.device)
    model = model.to(args.device)

    loss_fn = nn.MSELoss()
    val_loader = DataLoader(dataset=data, batch_size=args.bs, shuffle=False) # type: ignore
    print(f'{evaluate(model, val_loader, loss_fn, args.device):.3f}')


def train_ae(args):
    set_seed()

    data = read_gene_exp(args.genexp_file)

    if args.use_all:
        split = len(data)
    else:
        train_ids, test_ids = read_splits(args.splits_path)

    model = AE(args).to(args.device)
    if os.path.exists(args.save_path + 'model.pt'):
        load_model(model, args.save_path, device=args.device)
        print('Saved model found... resuming training')

    loss_fn = nn.MSELoss()

    optim = Adam(model.parameters(), lr=args.lr)

    if not args.use_all:
        train_data = [data[i] for i in train_ids] # type: ignore
        test_data = [data[i] for i in test_ids] # type: ignore
        train_loader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True) # type: ignore
        val_loader = DataLoader(dataset=test_data, batch_size=args.bs, shuffle=False) # type: ignore
    else:
        train_loader = DataLoader(dataset=list(data.values()), batch_size=args.bs, shuffle=True) # type: ignore
        val_loader = train_loader

    print(f'Train Size: {len(train_loader)}, Validation Size: {len(val_loader)}')
    model.train()

    best_score, best_epoch = np.inf, 0

    for epoch in range(1,args.epochs+1):
        batch_loss = []
        for batch in train_loader:
            batch = batch.to(args.device)
            recon = model(batch)

            loss = loss_fn(recon, batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            batch_loss.append(loss.item())

        print(f'Epoch = {epoch}, Train Loss = {np.mean(batch_loss):.3f}')

        # eval every 5 epochs
        if epoch % 5 == 0:
            ret = evaluate(model, val_loader, loss_fn, args.device)
            print(f'Validation Loss = {ret:.3f}')

            if best_score > ret:
                best_score, best_epoch = ret, epoch
                save_model(model, optim, args)

    #if not args.use_all:
    #    save_model(model, optim, args)
    print(f'Best model saved for epoch {best_epoch}')

    # just check loading
    load_model(model, args.save_path, args.device)
    print(f'{evaluate(model, val_loader, loss_fn, args.device):.3f}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--genexp_file', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ae_in_size', type=int, default=19177)
    parser.add_argument('--ae_out_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--splits_path', default='')
    parser.add_argument('--save_path', default='tmp/ae/')
    parser.add_argument('--use_all', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cuda', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    train_ae(args)

    test_ae(args)
