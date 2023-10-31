# Precision Anti-Cancer Drug Selection via Neural Ranking

Authors: Vishal Dey, Xia Ning


Workshop paper: Accepted in [BioKDD '23](https://biokdd.org/biokdd23/index.html)

Full version: In Progress

This repository provides the source code for the proposed methods: $\mathtt{Pair\text{-}PushC}$, $\mathtt{List\text{-}One}$ and $\mathtt{List\text{-}All}$ in our paper.

## Environments
Operating systems: Red Hat Enterprise Linux (RHEL) 7.7

Install packages under conda environments
```
conda create -n drugrank python=3.9
conda activate drugrank
conda install -y -c rdkit rdkit=2023.03.3
conda install -y numpy=1.26.0 scipy=1.9.1
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install torch=1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
````

## Datasets
- Download CCLE gene expression data 22Q1 version from [here](https://ndownloader.figshare.com/files/34008404) and save it as `data/CCLE/CCLE_expression.csv`.
- Please use the provided CTRP and PRISM datasets which are already processed. The processed datasets can be downloaded from [here](https://drive.google.com/drive/folders/1_w3_FSB0V4gzIdqku2enNfJIDeM5_pyO?usp=sharing). Unzip the `ctrp.zip` and `prism.zip` inside the `data` directory.
- For $\mathtt{pLETORg}$, we selected and used a set of M genes (out of 19,177 genes in CCLE gene expression data) using Elastic Net.
- TODO: add instructions to process data from scratch.

## Experiments

### For pre-training $\mathtt{GeneAE}$

- Check `scripts/run_ae.sh` on how to run the pretraining code in `src/train_ae.py`.
- Each pre-training run will take only a few minutes on a single V100 GPU.

### For ranking 
Run the below code to train $\mathtt{List\text{-}One}$ with default hyper-parameters

```
export DATA_FOLDER="data/ctrp/"
python src/cross_validate.py --model listone --data_path $DATA_FOLDER/LCO/aucs.txt --smiles_path $DATA_FOLDER/cmpd_smiles.txt --splits_path $DATA_FOLDER/LCO/pletorg/ --pretrained_ae -ae_path ${ae_path} -fgen morgan_count --setup LCO
```

Run the below code to train $\mathtt{List\text{-}All}$ with default hyper-parameters

```
export DATA_FOLDER="data/ctrp/"
python src/cross_validate.py --model listall --data_path $DATA_FOLDER/LCO/aucs.txt --smiles_path $DATA_FOLDER/cmpd_smiles.txt --splits_path $DATA_FOLDER/LCO/pletorg/ --pretrained_ae -ae_path ${ae_path} -fgen morgan_count -M 0.5 --setup LCO
```

Run the below code to train $\mathtt{Pair\text{-}PushC}$ with default hyper-parameters

```
export DATA_FOLDER="data/ctrp/"
python src/cross_validate.py --model pairpushc --data_path $DATA_FOLDER/LCO/aucs.txt --smiles_path $DATA_FOLDER/cmpd_smiles.txt --splits_path $DATA_FOLDER/LCO/pletorg/ --pretrained_ae -ae_path ${ae_path} -classc -fgen morgan_count --setup LCO
```
where ${ae_path} should be the path to the directory containing the saved models.

- `model` specifies the type of model to train.
- `data_path` specifies the file path containing final processed list of cell ID, drug ID and AUC values (comma-separated).
- `smiles_path` specifies the file path containing the list of tab-separated drug ID and its SMILES string, the SMILES string must be the last column in this file. 
- `splits_path` specifies the path to the directory containing the folds, where each fold is saved as a directory.
- `ae_path` specifies the path to the directory containing the pretrained $\mathtt{GeneAE}$ model.
- check `utils/args.py` for other hyper-parameters.
- Use `export DATA_FOLDER="data/prism/"` for all experiments on PRISM dataset. 
- change the `splits_path` to `$DATA_FOLDER/LRO/`, `data_path` to `$DATA_FOLDER/LRO/aucs.txt` and `setup=LRO` for the LRO experiments.

Check the following scripts for hyper-parameter grid-search and cross-validation:
- `scripts/run_listone.sh` for $\mathtt{List\text{-}One}$.
- `scripts/run_listall.sh` for $\mathtt{List\text{-}All}$.
- `scripts/run_pairpushc.sh` for $\mathtt{Pair\text{-}PushC}$.
