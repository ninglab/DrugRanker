import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class TCBBLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, gamma=0.1, margin=None):
        super(TCBBLoss, self).__init__()
        self.loss = nn.Softplus(beta=1, threshold=50)
        self.relu = nn.ReLU()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, diff, labels1, labels2, sign):
        y = np.array(labels1) == np.array(labels2)
        pn_pairs = (y == 0)   # for (sens, insens) or (insens, sens) pairs
        pp_pairs = (y == 1) & (np.array(labels1) == 1)   # for (sens,sens) pairs
        nn_pairs = (y == 1) & (np.array(labels1) == 0)   # for (insens,insens) pairs
        # sign should be 1 if (+,-) pair and -1 if (-,+) pair

        bloss = 0
        if sum(pp_pairs):
            bloss = self.alpha*torch.mean(self.loss(-sign[pp_pairs]*diff[pp_pairs]), dim=0)
        if sum(pn_pairs):
            bloss += (1-self.alpha-self.beta)*torch.mean(self.loss(-sign[pn_pairs]*diff[pn_pairs]), dim=0)
        if sum(nn_pairs):
            bloss += self.beta*torch.mean(self.loss(-sign[nn_pairs]*diff[nn_pairs]), dim=0)
        return bloss


class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, y_pred, y_true):
        pred_max = f.softmax(y_pred, dim=0) + 1e-9
        true_max = f.softmax(-y_true, dim=0)  # need to reverse the sign
        pred_log = torch.log(pred_max)
        return torch.mean(-torch.sum(true_max*pred_log))


class AttLoss(nn.Module):
    def __init__(self):
        super(AttLoss, self).__init__()

    def forward(self, y_pred, y_label):
        pred_max = f.softmax(y_pred/0.5, dim=1) + 1e-9
        pred_log = torch.log(pred_max)
        return torch.mean(-torch.sum(y_label*pred_log))
