import numpy as np
import torch
from utils.metrics import *
from utils.common import *
from features.featurization import mol2graph
from dataloader.loader import to_batchgraph

from collections import defaultdict
import time

def evaluate(clobj, model, test_dataloader, args, Kpos):
	model.eval()

	true_auc = []
	preds = []
	ccl_ids  = []
	cpd_ids = []
	labels   = []
	in_test = []

	if args.model in ['pairpushc', 'listone', 'listall']:
		for batch in test_dataloader:
			mols, features, clids = [], [], []

			for d in batch:
				true_auc.append(d.auc)
				mols.append(d.smiles)
				features.append(d.features)
				ccl_ids.append(d.clid)
				clids.append(d.clid)
				labels.append(d.label)
				in_test.append(d.in_test)

			cl_emb = torch.from_numpy(np.asarray(clobj.get_expression(clids))).to(args.device)
			cpd_ids += [d.cpdid for d in batch]
			
			molgraph = to_batchgraph(mols) if args.gnn else None

			pred = model(cl_emb, cmp1=molgraph, smiles1=mols, feat1=features, output_type=0).data.cpu().flatten().tolist()
			preds.extend(pred)
	else:
		raise ValueError(f'Model "{args.model}" not supported.')

	pred_dict = None
	metrics, m_clid, pred_dict = compute_metrics(true_auc, preds, ccl_ids, labels, in_test, cpd_ids, Kpos)
	return preds, true_auc, metrics, m_clid, pred_dict