import numpy as np
import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser

np.random.seed(123)
K = 5

drugs_per_cell = defaultdict(set)  # cells per drug
cells_per_drug = defaultdict(set)  # drugs per cell
set_cpds = set()
set_clids = set()
count_pcells = {}
aucs = defaultdict(list)

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, help='path to data directory')
parser.add_argument('--save_dir', type=str, help='path to save directory')
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.save_dir
splitter = '\t'

# first pass to get number of drugs per cell line
with open(data_dir+'final_list_auc.txt', 'r') as fp:
	next(fp)
	for line in fp.readlines():
		tmp = line.strip().split(splitter)
		if tmp[0] not in count_pcells:
			count_pcells[tmp[0]] = 0
		else:
			count_pcells[tmp[0]] += 1

# exclude the cell lines with less than 50 drugs tested in them
with open(data_dir+'final_list_auc.txt', 'r') as fp:
	next(fp)
	for line in fp.readlines():
		tmp = line.strip().split(splitter)
		if count_pcells[tmp[0]] < 50:
			continue
		cells_per_drug[tmp[1]].add(tmp[0])
		drugs_per_cell[tmp[0]].add(tmp[1])
		set_cpds.add(tmp[1])
		set_clids.add(tmp[0])			
		aucs[(tmp[0], tmp[1])].append(tmp[2])

# get the mean for each cell line and cpd
for k,v in aucs.items():
	aucs[k] = np.mean(list(map(float, v)))

# write the list of aucs to the file
fout = open(output_dir+'aucs.txt', 'w')
print('\n'.join([k[0]+splitter+k[1]+splitter+str(v) for k,v in aucs.items()]), file=fout)
fout.close()

cv = []

cpd_pc = defaultdict(set)

# for each drug, choose 1 cell line which have auc value
for cpd in set_cpds:
	choose_cell = np.random.choice(list(cells_per_drug[cpd]), 1)[0]
	cpd_pc[choose_cell].add(cpd)

#print([(k, len(v)) for k,v in cpd_pc.items()])

# for the rest of auc data per cell line, divide into K splits
# for the training, merge the splits with the previous added data
k_folds_pc = defaultdict(list)

print(len(cpd_pc), len(set_cpds), len(set_clids), np.mean([len(v) for v in cells_per_drug.values()]),
	  np.mean([len(v) for v in drugs_per_cell.values()]))

for cell in set_clids:
	# get all cpds for the cell line
	all_cpds = set([k[1] for k,v in aucs.items() if k[0] == cell])
	# remove the ones already considered
	remain = list(all_cpds - cpd_pc[cell])
	np.random.shuffle(remain)

	k_folds_pc[cell] = [set() for _ in range(K)]

	for i, cpd in enumerate(remain):
		k_folds_pc[cell][i%K].add(cpd)


## now create training and testing splits by merging
list_cpds = list(set_cpds)

#fout = open(output_dir+'drugs.txt', 'w')
#print('\n'.join(list_cpds), file=fout)
#fout.close()

#fout = open(output_dir+'cells.txt', 'w')
#print('\n'.join(list(set_clids)), file=fout)
#fout.close()

fout = open(output_dir+'splits.txt', 'w')
output_sep = '|'
for cell in set_clids:
	for k in range(K):
		# create train, val and test splits
		val_split = list(k_folds_pc[cell][k])
		test_split = list(k_folds_pc[cell][(k+1)%K])
		mask = np.ones(K, dtype=bool)
		mask[k] = mask[(k+1)%K] = False
		train_split = list(set.union(*(np.array(k_folds_pc[cell])[mask])))

		# add the selected ones
		train_split.extend(list(cpd_pc[cell]))

		print(cell + '\t' + str(k) + '\t'
			+ output_sep.join(map(str, train_split)) + '\t' 
			+ output_sep.join(map(str, val_split)) + '\t' 
			+ output_sep.join(map(str, test_split)), file=fout)

fout.close()
