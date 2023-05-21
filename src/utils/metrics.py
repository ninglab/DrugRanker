import numpy as np
from collections import defaultdict
from sklearn.metrics import ndcg_score, mean_squared_error
import itertools


"""
In this implementation, ground truth scores are AUC values -- lower indicates higher sensitivity
Predicted scores are either difference (s1-s2) OR scores of individual compounds; 
where higher values indicate s1 more sensitive than s2 OR comp is more sensitive.
"""


def mse(pred, actual):
    return mean_squared_error(pred, actual)

def rmse(pred, actual):
    return np.sqrt(mean_squared_error(pred, actual)) # type: ignore


def compute_lCI(pred_scores, true_auc, labels):
    """
    computes percentage of correctly ordered pairs where
    sensitive compounds are ranked above insensitive ones
    """
    correct = 0
    n_pairs = 0

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i !=j and labels[i] > labels[j]: # sensitive i and insensitive j
                n_pairs += 1
                if pred_scores[i] >= pred_scores[j]:
                    correct += 1
    if n_pairs == 0:
        return np.nan
    return correct/n_pairs


def compute_CI(pred_scores, true_auc):
    """ computes CI by def """
    correct = 0
    n_pairs = 0

    for i in range(len(pred_scores)):
        for j in range(len(pred_scores)):
            # in ground truth, lower values mean higher sensitivity; reverse in predictions
            if i!=j and true_auc[i] <= true_auc[j]:
                n_pairs += 1
                if pred_scores[i] >= pred_scores[j]:
                    correct += 1
    if n_pairs == 0:
        return np.nan
    return correct/n_pairs


def compute_sCI(pred_scores, true_auc, labels):
    """ considers CI only with pairs of sensitive comp, by def of equ (4) """
    correct = 0
    n_pairs = 0

    for i in range(len(labels)):
        for j in range(len(labels)):
            # only consider ranks within sensitive
            if (i!=j) and (labels[i] + labels[j] == 2) \
                and true_auc[i] <= true_auc[j]:
                n_pairs += 1
                if pred_scores[i] >= pred_scores[j]:
                    correct += 1

    if n_pairs == 0:
        return np.nan
    return correct/n_pairs


def compute_recall(pred_scores, true_ic, K=10):
    assert(K >= 1)
    true_comp_rank = np.argsort(true_ic, kind='mergesort')[:K]      # smaller values => ranked at top
    pred_comp_rank = np.argsort(pred_scores, kind='mergesort')[::-1][:K]  # higher values => ranked at top
    return sum(pred_comp_rank == true_comp_rank)/K


def compute_recall_topk(pred_scores, true_ic, K=10):
    assert(len(pred_scores) == len(true_ic))
    k = max(1, int(np.ceil(K/100.0*len(pred_scores)))) # type: ignore
    assert(k >= 1)
    return compute_recall(pred_scores, true_ic, k)


def compute_ndcg(pred_scores, true_ic, K=10):
    # get relevance scores from AUC values: highest AUC gets 0 relevance; lowest AUC gets highest relevance
    true_rel = np.argsort(np.argsort(true_ic)[::-1]) # higher relevance to lower scores
    return ndcg_score(np.array([true_rel]), np.array([pred_scores]), k=K)

def compute_ndcg_topk(pred_scores, true_ic, K=10):
    assert(len(pred_scores) == len(true_ic))
    k = max(1, int(np.ceil(K/100.0*len(pred_scores)))) # type: ignore
    assert(k >= 1)
    return compute_ndcg(pred_scores, true_ic, k)


def compute_hit(pred_ranks, labels, K=10):
    ## equ 12
    return sum(np.array(labels)[pred_ranks[:K]])


def compute_AT(pred_ranks, true_ranks, labels, K=10):
    sens = np.where(np.array(labels) == 1)[0]
    top_k_true = set(true_ranks[:K]).intersection(set(sens))
    top_k_pred = set(pred_ranks[:K])

    K = min(K, len(top_k_true))
    # for test set, K can be 0
    if K == 0:
        return np.nan
    return len(top_k_true.intersection(top_k_pred))/K  # equ (13)


def compute_NT(pred_ranks, true_ranks, in_test, labels, K=10):
    in_test = np.where(np.array(in_test) == 1)[0]
    sens = np.where(np.array(labels) == 1)[0]
    top_k_true = set(true_ranks[:K]).intersection(set(sens)).intersection(set(in_test)) # new drugs that should be among top-k sensitive drugs in the ground truth
    top_k_pred = set(pred_ranks[:K])
    top_k_pred_in_true = top_k_true.intersection(top_k_pred)
    if len(top_k_true) == 0:
        return np.nan
    return len(top_k_pred_in_true)/len(top_k_true)# equ (14)


def compute_AP(pred_ranks, labels, K=10):
    ap = 0
    deno = compute_hit(pred_ranks, labels, K)
    if deno == 0:
        return 0
    # eq 10
    for j in range(1, K+1):
        # compute precision for every kth position (1..K)
        ap += compute_precision(pred_ranks, labels, j)*labels[pred_ranks[j-1]]

    return ap/deno

def compute_precision(pred_ranks, labels, K=10):
    # among the top-K predicted ranked compounds, how many of them are actually sensitive
    topk_ind = pred_ranks[:K]
    return np.mean(np.array(labels)[topk_ind])  # equ 11 in TCBB paper; ratio of sensitive among top-K ranked


def estimate_score(scores_cpd):
    est_score = []
    for c in scores_cpd:
        est_score.append(np.mean(c))

    return est_score

def _compute_metrics(y_true_pc, y_pred_pc, label_pc, in_test_pc, flag_comb, Kpos):
    ci, lci, sci = [], [], []
    ah = defaultdict(list)
    ap = defaultdict(list)
    ndcg = defaultdict(list)
    at = defaultdict(list)
    nt = defaultdict(list)

    dict_metrics = {}

    for k, y_pred in y_pred_pc.items():
        #y_pred = y_pred_pc[k]
        y_true = y_true_pc[k]
        label = label_pc[k]
        dict_metrics[k] = {}

        # indices of ranked elements 
        # (value at 'i' indicates index of element in y_pred that should be ranked at i)
        # higher pred scores y_pred indicate more sensitive
        # lower true AUC y_true indicate more sensitive
        pred_ranks = np.argsort(y_pred, kind='mergesort')[::-1]
        true_ranks = np.argsort(y_true, kind='mergesort')

        dict_metrics[k]['CI'] = compute_CI(y_pred, y_true)
        ci.append(dict_metrics[k]['CI'])
        dict_metrics[k]['lCI'] = compute_lCI(y_pred, y_true, label)
        lci.append(dict_metrics[k]['lCI'])
        dict_metrics[k]['sCI'] = compute_sCI(y_pred, y_true, label)
        sci.append(dict_metrics[k]['sCI'])

        for Kp in Kpos:
            dict_metrics[k][f'AP@{Kp}'] = compute_AP(pred_ranks, label, K=Kp)
            ap[Kp].append(dict_metrics[k][f'AP@{Kp}'])
            dict_metrics[k][f'AH@{Kp}'] = compute_hit(pred_ranks, label, K=Kp)
            ah[Kp].append(dict_metrics[k][f'AH@{Kp}'])
            dict_metrics[k][f'NDCG@{Kp}'] = compute_ndcg(y_pred, y_true, K=Kp)
            ndcg[Kp].append(dict_metrics[k][f'NDCG@{Kp}'])

            if flag_comb:
                dict_metrics[k][f'AT@{Kp}'] = compute_AT(pred_ranks, true_ranks, label, K=Kp)
                at[Kp].append(dict_metrics[k][f'AT@{Kp}'])
                dict_metrics[k][f'NT@{Kp}'] = compute_NT(pred_ranks, true_ranks, in_test_pc[k], label, K=Kp)
                nt[Kp].append(dict_metrics[k][f'NT@{Kp}'])

    ret = {'CI' : np.nanmean(ci), 'lCI': np.nanmean(lci), 'sCI': np.nanmean(sci)}
    for k in Kpos:
        ret[f'AP@{k}'] = np.nanmean(ap[k])
        ret[f'AH@{k}'] = np.nanmean(ah[k])
        ret[f'NDCG@{k}'] = np.nanmean(ndcg[k])
    if flag_comb:
        for k in Kpos:
            ret[f'AT@{k}'] = np.nanmean(at[k])
            ret[f'NT@{k}'] = np.nanmean(nt[k])

    return ret, dict_metrics

"""
def compute_metrics_from_pairs(mol_pairs, true_auc, pred, ccl_ids, labels, plabels, in_test, \
                                margin=0.5, pred_type=2):
    # create list of ranked pairs based on diff
    mols_pc = defaultdict(set)

    # first create an ordered list of mols per cell line -- required for indexing
    for pair, clid in zip(mol_pairs, ccl_ids):
        mols_pc[clid].add(pair[0])
        mols_pc[clid].add(pair[1])

    y_true_pc = defaultdict(list)
    y_pred_pc = defaultdict(list)
    y_scores_pc = defaultdict(list)
    in_test_pc = defaultdict(list)
    label_pc  = defaultdict(list)
    pmat_pc = {}
    est_score_pc = {}

    correct_pairs_pc = dict.fromkeys(list(set(ccl_ids)),0)
    total_pairs_pc = dict.fromkeys(list(set(ccl_ids)), 0)

    for k,v in mols_pc.items():
        mols_pc[k] = list(v)
        y_true_pc[k] = [0]*len(v)
        if pred_type == 1:
            for _ in range(len(v)):
                y_scores_pc[k].append(list())
        in_test_pc[k] = [0]*len(v)
        label_pc[k] = [0]*len(v)
        pmat_pc[k] = np.zeros((len(v),len(v)))

    for i in range(len(mol_pairs)):
        pair = mol_pairs[i]
        y_true, y_pred, label, tlb = true_auc[i], pred[i], labels[i], in_test[i]
        clid = ccl_ids[i]
        if plabels:
            plabel = plabels[i]
        ## ignore most uncertain predictions 
        #if abs(y_diff) < margin:

        if label[0] != label[1]:
            total_pairs_pc[clid] += 1
            # create list of pairs where the order of pairs is correct
            # if y_pred > 0 and diff in true rank < 0 OR y_pred < 0 and diff in true rank > 0
            if pred_type == 2:
                if y_pred*(y_true[0]-y_true[1]) < 0:
                    correct_pairs_pc[clid] += 1
            elif pred_type == 1:
                if (y_pred[0]-y_pred[1])*(y_true[0]-y_true[1]) < 0:
                    correct_pairs_pc[clid] += 1

        #if plabel < 0.5:
        #if label[0] == label[1]:
        #   continue

        # store the true AUC values in the order of compounds indexed
        i, j = mols_pc[clid].index(pair[0]), mols_pc[clid].index(pair[1])
        y_true_pc[clid][i] = y_true[0]
        y_true_pc[clid][j] = y_true[1]
        label_pc[clid][i] = label[0]
        label_pc[clid][j] = label[1]
        in_test_pc[clid][i] = tlb[0]
        in_test_pc[clid][j] = tlb[1]

        if pred_type == 2:
            pmat_pc[clid][i][j] = 1/(1+np.exp(-max(50,y_pred)))
            pmat_pc[clid][j][i] = 1/(1+np.exp(max(50,y_pred)))
            if pmat_pc[clid][i][j] > 0.5:
                y_pred_pc[clid].append((i, j))
            elif pmat_pc[clid][i][j] < 0.5:
                y_pred_pc[clid].append((j, i))

        elif pred_type == 1:
            y_scores_pc[clid][i].append(y_pred[0])
            y_scores_pc[clid][j].append(y_pred[1])

    # retrieve ranking for each cell line
    for cl in set(ccl_ids):
        if pred_type == 1:
            est_score_pc[cl] = estimate_score(y_scores_pc[cl])
        if total_pairs_pc[cl] == 0:
            correct_pairs_pc[cl] = 0
        else:
            correct_pairs_pc[cl] /= total_pairs_pc[cl]

    flag_comb = (sum(in_test_pc[clid]) != 0)
    metrics, m_clid = _compute_metrics(y_true_pc, est_score_pc, label_pc, in_test_pc, flag_comb)
    return metrics, m_clid, np.nanmean(list(correct_pairs_pc.values()))
"""

def compute_metrics(true_auc, pred_scores, ccl_ids, labels, in_test, cpd_ids, Kpos):

    in_test_pc = defaultdict(list)
    y_true_pc = defaultdict(list)
    y_pred_pc = defaultdict(list)
    label_pc  = defaultdict(list)
    pred_pc = {}
    for cl_id in ccl_ids:
        pred_pc[cl_id] = {}

    for y_true, y_pred, l, tlb, clid, cpid in zip(true_auc, pred_scores, labels, in_test, ccl_ids, cpd_ids):
        y_true_pc[clid].append(y_true)
        y_pred_pc[clid].append(y_pred)
        label_pc[clid].append(l)
        in_test_pc[clid].append(tlb)
        pred_pc[clid][cpid] = y_pred

    # flag_comb denotes the combination in the loader: if purely test or train+test compounds
    flag_comb = (sum(in_test_pc[ccl_ids[0]]) != 0)
    return *_compute_metrics(y_true_pc, y_pred_pc, label_pc, in_test_pc, flag_comb, Kpos), pred_pc  # python >= 3.8 required



########################################################################
########################################################################
if __name__ == '__main__':
    x = [.12, 1, 3, .4, 6, 8, .1, 4, 9, 10]
    y = [24, 1, 4, 12, 3, 9, 30, 22, 11, 8]
    labels = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]
    in_test = np.array([True,False,False,False,True,True,False,False,True,False])
    #print(compute_metrics(x, y, ['A']*len(x), labels, in_test))
    pmat = np.zeros((10, 10))
    for i, j in itertools.permutations(range(10), r=2):
        pmat[i][j] = (j + 1) / (i + j + 2)

    #print(pmat)
    # Estimating Bradley-Terry model parameters.
    #y = mle(pmat)
    pred_ranks = np.argsort(y, kind='mergesort')[::-1]
    true_ranks = np.argsort(x, kind='mergesort')
    print(compute_CI(y,x))
    print(compute_sCI(y,x,labels))
    print(compute_lCI(y,x,labels))
    print(compute_hit(pred_ranks, labels, 5))
    print(compute_hit(pred_ranks, labels, 10))
    print(compute_precision(pred_ranks, labels, 5))
    print(compute_precision(pred_ranks, labels, 10))
    print(compute_AP(pred_ranks, labels, 5))
    print(compute_AP(pred_ranks, labels, 10))
    print(compute_AT(pred_ranks, true_ranks, 2))
    print(compute_NT(pred_ranks, true_ranks, in_test, 2))
