# Creating processed datasets

## Experimental Setup
- Setup1: Leave-responses-out (LRO) cross validation
- Setup2: Leave-cell lines-out (LCO) cross validation (Unseen cell lines with 1/5th cell lines for each cancer type hold out during testing)
- Below, we provide detailed instructions and commands on how to reproduce the provided processed and cleaned data files from source files.
- Each setup requires different type of data split as discussed below.

## Get gene expression data
- CCLE: Download CCLE expression data from [here](https://ndownloader.figshare.com/files/34008404) and save it as `CCLE/CCLE_expression.csv`.
- Combined: Download the `combined_rnaseq_data` file with gene expression data for cell lines renamed for CTRPv2 from [here](https://modac.cancer.gov/assetDetails?dme_data_id=NCI-DME-MS01-8088592) and save it in `Combined/`.

## Preprocess data from source files
- To create the processed data files, first download the source datasets as mentioned below.
- For CTRPv2 with adjusted AUCs, we download the following files from [here](https://modac.cancer.gov/assetDetails?dme_data_id=NCI-DME-MS01-8088592):
	- For drug information: `drug_info` 
	- For adjusted AUCs: `combined_single_response_agg`
- For PRISM, we download the following files from [here](https://depmap.org/portal/download/all/?releasename=PRISM+Repurposing+19Q4)
	- For drug information and AUCs: `secondary-screen-dose-response-curve-parameters.csv`
- Run the commands provided in `commands_ctrpv2.log` and `commands_prism.log` to clean the files within respective directories for the following steps.

## For LRO Setup
```
mkdir -p ctrpv2/LRO/
python scripts/create_cv.py --data_dir /fs/ess/PCON0041/Vishal/DrugRank/data/Combined/ --save_dir ctrpv2/LRO/
```
The above script creates the list of AUCs in `aucs.txt` and splits in `splits.txt`. Run this command similarly for PRISM.

### To create setup for pLETORg:
```
python scripts/create_setup_LRO.py --data_dir ctrpv2/LRO/ --save_dir ctrpv2/LRO/pletorg/ --genexp_file Combined/combined_rnaseq_data_combat 
```
- The above script does the following:
	- Creates the full cell line - drug sensitivity matrix and saves as `auc_matrix.txt`. M x N matrix
	- Saves the list of cell IDs as `cells.txt`. # cells = M
	- Saves the list of drug IDs as `drugs.txt`. # drugs = N
	- Note that the order of cells across rows and the order of drugs across columns must be consistent
		with the respective ordering in the files `cells.txt` and `drugs.txt`.
- Rerun the script accordingly for PRISM.

### To select informative genes using Elastic Net
```
python scripts/lasso.py --data_dir ctrpv2/LRO/ --l1_ratio 0.5
python scripts/lasso.py --data_dir prism/LRO/ --l1_ratio 0.1
```
- This runs Elastic Net on the drug response matrix to identify informative genes for each cell line, following the standard setup in pLETORg.

### To compute pairwise cell line similarity using the selected genes as features
- To compute cosine similarities:
```
python scripts/cell_similarity.py --data_dir ctrpv2/LRO/ --genes ctrpv2/LRO/genes.txt --sim cos
python scripts/cell_similarity.py --data_dir prism/LRO/ --genes prism/LRO/genes.txt --sim cos
```
- To compute simliarities using RBF kernel:
```
python scripts/cell_similarity.py --data_dir ctrpv2/LRO/ --genes ctrpv2/LRO/genes.txt --sim rbf
python scripts/cell_similarity.py --data_dir prism/LRO/ --genes prism/LRO/genes.txt --sim rbf
```

## For LCO Setup
```
python scripts/create_setup_LCO.py --data_dir ctrpv2/LRO/ --save_dir ctrpv2/LCO/pletorg/
python scripts/create_setup_LCO.py --data_dir prism/LRO/ --save_dir prism/LCO/pletorg/
```
- Creates the auc matrices separately for training, val and test in each fold
- Saves the list of cell IDs as `*_cells.txt` and the union of cells in `../cells.txt`, and corresponding pairwise similarity matrix
- Note that the order of cells across rows is consistent with the respective ordering in the similarity matrix file.