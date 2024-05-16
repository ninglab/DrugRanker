# get records with non-empty DepMap IDs, drug name and AUC
##awk -vFPAT='([^,]*)|("[^"]+")' -vOFS='\t' '$1 != "" {print $1,$3,$(NF-3)}' prism_ic50.csv > depmapid_cmp_auc.txt
awk -vFPAT='([^,]*)|("[^"]+")' -vOFS='\t' '$1 != "" {print $2,$12,$9}' secondary-screen-dose-response-curve-parameters.csv > depmapid_cmp_auc.txt


# compound SMILES mapping
awk -v FPAT='([^,]*)|("[^"]*")' '{print($12"\t"$17)}' secondary-screen-dose-response-curve-parameters.csv | sort | uniq | awk -F'\t' '{split($2, arr, ","); print $1 "\t" arr[length(arr)]}' | tr -d '"' > cmpd_smiles.txt
##less prism_drugs.csv | grep "," | gawk -vFPAT='([^,]*)|("[^"]+")' '{print $1"\t"$2}' > cmpd_smiles.txt

# compound MOA mapping
awk -v FPAT='([^,]*)|("[^"]*")' '{print($12"\t"$13)}' secondary-screen-dose-response-curve-parameters.csv | sort | uniq > cmpd_moa.txt

# 4 DepMapIDs have missing expression data -- need to discard them
printf "ACH-000047\nACH-000309\nACH-000979\nACH-001024\n" > ids_not_in_ccle
##awk 'NR==FNR {a[$1];next} {if(!($1 in a)) print $1}' /fs/ess/PCON0041/Vishal/DrugRank/data/test/ids_in_ccle.txt <(sed 1d depmapid_cmp_auc.txt | awk -F'\t' '{print $1}' | sort | uniq) > ids_not_in_ccle.txt

# 2 compounds do not have corresponding SMILES information
awk -F'\t' '{print $1}' cmpd_smiles.txt | sort | uniq) <(awk -F'\t' '{print $2}' final_list_auc.txt | sort | uniq) > missing_smiles.txt

grep -vFf <(cat ids_not_in_ccle.txt missing_smiles.txt) depmapid_cmp_auc.txt > final_list_auc.txt