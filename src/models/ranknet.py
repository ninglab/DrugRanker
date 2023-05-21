import torch
import pdb
import numpy as np
import torch.nn as nn

from models.mpn import MPNN
from models.ae import AE
from torch_geometric.nn import MLP
from utils.common import load_model
from itertools import chain


class Fingerprint(nn.Module):
    def __init__(self, args):
        super(Fingerprint, self).__init__()
        input_dim = 0
        if args.feature_gen == 'rdkit_2d' or args.feature_gen=='rdkit_2d_normalized':
            input_dim = 200
        elif args.feature_gen == 'morgan' or args.feature_gen == 'morgan_count':
            input_dim = 2048

        self.ffn1 = nn.Linear(input_dim, 128)
        self.ffn2 = nn.Linear(128, args.mol_out_size)
        #self.mlp = MLP(channel_list=[input_dim, 256, 128, args.mol_out_size])
        self.relu = nn.ReLU()
        self.device = args.device

    def forward(self, molgraph, features):
        features = torch.from_numpy(np.stack(features)).float().to(self.device)
        #return self.mlp(features)
        return self.ffn2(self.relu(self.ffn1(features)))


class Scoring(nn.Module):
    def __init__(self, args):
        super(Scoring, self).__init__()
        self.out_size = args.mol_out_size

        if args.feature_gen:
            if not args.use_features_only:
                self.out_size += 2048

        self.scoring = args.scoring
        if args.update_emb == 'cell+list-attention2':
            self.scoring = 'mlp2'
            self.ffn = MLP(in_channels=self.out_size ,
                                    hidden_channels=25, num_layers=2, out_channels=1)
        elif args.scoring == 'linear':
            self.ffn = nn.Linear(args.ae_out_size, self.out_size)
        elif args.scoring == 'mlp':
            self.ffn = MLP(in_channels=self.out_size+args.ae_out_size ,
                                    hidden_channels=25, num_layers=2, out_channels=1)


    def forward(self, cell_emb, cmp1_emb, cmp2_emb=None, output_type=2):
        """ Type 2: outputs the difference of predicted AUCs (if two input embeddings)
            Type 1: outputs paired scores for paired compounds
            Type 0: only list of scores for list of compounds
        """
        if output_type == 2:
            if self.scoring == 'linear':
                return (self.ffn(cell_emb)*(cmp1_emb - cmp2_emb)).sum(dim=1)
            elif self.scoring == 'mlp':
                return self.ffn(torch.concat((cell_emb, cmp1_emb), dim=1)).squeeze() - \
                self.ffn(torch.concat((cell_emb, cmp2_emb), dim=1)).squeeze()
            #score = (self.ffn(cmp1_emb - cmp2_emb)*cell_emb).sum(dim=1)
        elif output_type == 1:
            if self.scoring == 'linear':
                score1 = (self.ffn(cell_emb)*cmp1_emb).sum(dim=1)
                score2 = (self.ffn(cell_emb)*cmp2_emb).sum(dim=1)
            elif self.scoring == 'mlp':
                score1 = self.ffn(torch.concat((cell_emb, cmp1_emb), dim=1)).squeeze()
                score2 = self.ffn(torch.concat((cell_emb, cmp2_emb), dim=1)).squeeze()
            return score1, score2
        else:
            if self.scoring == 'linear':
                score = (self.ffn(cell_emb)*cmp1_emb).sum(dim=1)
            elif self.scoring == 'mlp':
                score = self.ffn(torch.concat((cell_emb, cmp1_emb), dim=1)).squeeze()
            elif self.scoring == 'mlp2':
                score = self.ffn(cmp1_emb).squeeze()
            #score = (self.scoring(cmp1_emb)*cell_emb).sum(dim=1)
            return score

def sim(x1, x2, sigma=1, kernel='l2'):
    if kernel == 'l2':
        return torch.sum((x1-x2)**2, dim=1)
    if kernel == 'rbf':
        return torch.exp(-sigma*torch.sum((x1-x2)**2, dim=1))


class RankNet(nn.Module):
    def __init__(self, args, mode=None):
        super(RankNet, self).__init__()
        self.enc_type = args.gnn

        if args.feature_gen:
            self.enc = Fingerprint(args)
            self.enc_type = args.feature_gen
        elif args.gnn == 'dmpn':
            self.enc = MPNN(args)
            ## over-ride args.mol_out_size
            if 'hier-cat' in args.pooling:
                args.mol_out_size *= args.message_steps
        else:
            raise NotImplementedError

        # for now only MLP update
        self.update_emb = args.update_emb

        self.classify_pairs = args.classify_pairs
        self.classify_cmp = args.classify_cmp
        self.cluster = args.cluster
        self.agg_emb = args.agg_emb

        if self.update_emb == 'concat':
            self.u_mlp = MLP(channel_list=args.mol_out_size*2+args.ae_out_size,
                             hidden_channels=50, num_layers=2, out_channels=args.mol_out_size)
        elif self.update_emb in ['cell-attention', 'sum', 'cell+list-attention']:
            self.u_mlp = MLP(in_channels=args.mol_out_size+args.ae_out_size,
                             hidden_channels=50, num_layers=2, out_channels=args.mol_out_size)

        ## override the mol_out_size again if update rule is concatenation
        if self.update_emb!='None' and args.agg_emb == 'concat':
            args.mol_out_size *= 2

        if self.classify_pairs:
            self.classifierp = MLP(in_channels=args.mol_out_size*2+args.ae_out_size,
                                    hidden_channels=25, num_layers=2, out_channels=1)
        if self.classify_cmp:
            self.classifierc = MLP(in_channels=args.mol_out_size+args.ae_out_size,
                                    hidden_channels=25, num_layers=2, out_channels=1)

        self.ae = AE(args)

        if self.update_emb == 'list-attention':
            self.mha = nn.MultiheadAttention(args.mol_out_size, 4)
        elif 'cell+list-attention' in self.update_emb:
            te_layer = nn.TransformerEncoderLayer(args.mol_out_size, 5, 128)
            self.te = nn.TransformerEncoder(te_layer, 1)

        if args.pretrained_ae:
            load_model(self.ae, args.trained_ae_path, args.device)

        self.scoring = Scoring(args)

    def update(self, cell_emb, cmp1_emb, cmp2_emb=None):
        if self.update_emb == 'concat':
            x = torch.concat((cell_emb, cmp1_emb, cmp2_emb), dim=1) + torch.concat((cell_emb, cmp2_emb, cmp1_emb), dim=1)
            c = self.u_mlp(x)
        elif self.update_emb == 'sum':
            x = torch.concat((cell_emb, cmp1_emb+cmp2_emb), dim=1)
            c = self.u_mlp(x)
        

        elif 'cell' in self.update_emb:
            fused = self.u_mlp(torch.concat((cell_emb, cmp1_emb), dim=1))
            if self.update_emb == 'cell+list-attention':
                cmp1_emb = fused
            elif self.update_emb == 'cell-attention':
                return cmp1_emb

        if 'cell+list-attention' in self.update_emb:
            return self.te(cmp1_emb.unsqueeze(dim=1)).squeeze(dim=1)
        elif 'list-attention' in self.update_emb:
            output, weights = self.mha(cell_emb, cmp1_emb, cmp1_emb)
            return output

        gate1 = torch.sigmoid(c*cmp1_emb)
        gate2 = torch.sigmoid(c*cmp2_emb)
        if self.agg_emb == 'sum':
            cmp1_emb = (1+gate1)*cmp1_emb
            cmp2_emb = (1+gate2)*cmp2_emb
        elif self.agg_emb == 'concat':
            cmp1_emb = torch.concat((cmp1_emb, cmp1_emb*gate1), dim=1)
            cmp2_emb = torch.concat((cmp2_emb, cmp2_emb*gate2), dim=1)
        return cmp1_emb, cmp2_emb


    def forward(self, clines, cmp1=None, smiles1=None, feat1=None, \
                cmp2=None, smiles2=None, feat2=None, \
                pos=None, neg=None, output_type=2):
        cell_emb = self.ae(clines.float(), use_encoder_only=True)
        plabel = None
        clabel = None
        cmp_sim = None

        if output_type != 0:
            # outputs difference of scores or paired
            if pos and neg:
                cmp_emb = self.enc(cmp1, feat1)
                cmp1_emb = cmp_emb[pos]
                cmp2_emb = cmp_emb[neg]
            else:
                cmp1_emb = self.enc(cmp1, feat1)
                cmp2_emb = self.enc(cmp2, feat2)

            if self.update_emb != 'None':
                cmp1_emb, cmp2_emb = self.update(cell_emb, cmp1_emb, cmp2_emb)

            if self.classify_pairs:
                plabel = self.classifierp(torch.concat((cell_emb, cmp1_emb, cmp2_emb),dim=1)).squeeze()

            if self.classify_cmp:
                clabel1 = self.classifierc(torch.concat((cell_emb, cmp1_emb),dim=1)).squeeze()
                clabel2 = self.classifierc(torch.concat((cell_emb, cmp2_emb),dim=1)).squeeze()
                clabel = torch.concat((clabel1, clabel2))

            if self.cluster:
                cmp_sim = sim(cmp1_emb, cmp2_emb)

            return self.scoring(cell_emb, cmp1_emb, cmp2_emb, output_type), plabel, clabel, cmp_sim

        else:
            cmp_emb = self.enc(cmp1, feat1)

            if self.update_emb != 'None':
                cmp_emb = self.update(cell_emb, cmp_emb)
            return self.scoring(cell_emb, cmp_emb, output_type=output_type)

