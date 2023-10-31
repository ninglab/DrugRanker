# Experimental Setup
- Setup1: Leave-responses-out (LRO) cross validation
- Setup2: Leave-cell lines-out (LCO) cross validation (Unseen cell lines with 1/5th cell lines for each cancer type hold out during testing)

## Get gene expression data
- CCLE: Download CCLE expression data from [here](https://ndownloader.figshare.com/files/34008404) and save it as `CCLE/CCLE_expression.csv`.
- Combined: Download the `combined_rnaseq_data` file with gene expression data for cell lines renamed for CTRPv2 from [here](https://modac.cancer.gov/assetDetails?dme_data_id=NCI-DME-MS01-8088592) and save it in `Combined/`.

## To prepare data for training and testing for LRO Setup
- To create the processed data files, follow the below steps. 
- For CTRPv2 with adjusted AUCs, we download the following files from [here](https://modac.cancer.gov/assetDetails?dme_data_id=NCI-DME-MS01-8088592):
	- For drug information: `drug_info` 
	- For adjusted AUCs: `combined_single_response_agg`
- For PRISM, we download the following files from [here](https://depmap.org/portal/download/all/?releasename=PRISM+Repurposing+19Q4)
	- For drug information: `secondary-screen-dose-response-curve-parameters.csv`
	- For AUCs: `secondary-screen-dose-response-curve-parameters.csv`

mkdir -p ctrpv2/LRO/
python scripts/create_cv.py --data_dir /fs/ess/PCON0041/Vishal/DrugRank/data/Combined/ --save_dir ctrpv2/LRO/
```
The above file creates the list of AUCs in `aucs.txt` and splits in `splits.txt`.

### To create setup for pLETORg:
```
python scripts/create_setup_LRO.py --data_dir ctrpv2/LRO/ --save_dir ctrpv2/LRO/pletorg/ --genexp_file Combined/combined_rnaseq_data_combat 
```
- Creates the full cell line - drug sensitivity matrix and saves as `auc_matrix.txt`. M x N matrix
- Saves the list of cell IDs as `cells.txt`. # cells = M
- Saves the list of drug IDs as `drugs.txt`. # drugs = N
- Note that the order of cells across rows and the order of drugs across columns must be consistent
	with the respective ordering in the files `cells.txt` and `drugs.txt`.

### To select informative genes using Elastic Net (**ADDED Post-Review**)
```
python scripts/lasso.py --data_dir ctrpv2/LRO/ --l1_ratio 0.5
python scripts/lasso.py --data_dir prism/LRO/ --l1_ratio 0.1
```

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

### To create LCO setup
```
python scripts/create_setup_LCO.py --data_dir ctrpv2/LRO/ --save_dir ctrpv2/LCO/pletorg/
python scripts/create_setup_LCO.py --data_dir prism/LRO/ --save_dir prism/LCO/pletorg/
```
- Creates the auc matrices separately for training, val and test in each fold
- Saves the list of cell IDs as `*_cells.txt` and the union of cells in `../cells.txt`, and corresponding pairwise similarity matrix
- Note that the order of cells across rows is consistent with the respective ordering in the similarity matrix file.


## Some additional steps to clean drug names manually
```
awk -F'\t' 'NR==FNR {a[tolower($1)]=$NF;  a[tolower($2)]=$NF; next} {if(!(tolower($2) in a) && !(tolower($3) in a)) print($1"\t"tolower($2)"\t"tolower($3)"\t"a[tolower($2)]"\t"a[tolower($3)]"\t"$4)}' broad_drug_info.txt drug_info | grep CTRP > drug_not_mapped.txt

awk -F'\t' 'NR==FNR {a[tolower($1)]=$NF;  a[tolower($2)]=$NF; next} {if((tolower($2) in a) || (tolower($3) in a)) print($1"\t"tolower($2)"\t"tolower($3)"\t"a[tolower($2)]"\t"a[tolower($3)]"\t"$4)}' broad_drug_info.txt drug_info | grep CTRP | tr -d '\r' > drug_mapped.txt

awk -F'\t' 'NR==FNR {a[$1];next} {if(!($1 in a)) print}' <(cat drug_mapped.txt drug_not_mapped.txt) drug_info  | grep CTRP > drug_left.txt

echo $'id\tsmiles' > ~/DrugRank/clean/data/ctrpv2/cmpd_smiles.txt 
cat drug_mapped.txt drug_not_mapped.txt drug_left.txt | awk -F'\t' '{if($4=="") {print($1"\t"$NF)} else {print($1"\t"$4)}}' >> ~/DrugRank/clean/data/ctrpv2/cmpd_smiles.txt
```