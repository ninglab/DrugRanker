# Cell line mapping
less cl_mapping | grep CCLE | grep CTRP | tr '\t' ',' > stripped_cell_name.txt

awk -F, 'NR==FNR {a[$1]=$2;next} {if("CCLE."$3 in a) print $1"\tCTRP."$3"\t"a["CCLE."$3]}' stripped_cell_name.txt ../CCLE/sample_info.csv > mapped_depmap_cells.txt

less combined_rnaseq_data_combat | grep CTRP | awk -F '\t' '{print $1}' > cells_with_gexp.txt


# Clean drug names manually
awk -F'\t' 'NR==FNR {a[tolower($1)]=$NF;  a[tolower($2)]=$NF; next} {if(!(tolower($2) in a) && !(tolower($3) in a)) print($1"\t"tolower($2)"\t"tolower($3)"\t"a[tolower($2)]"\t"a[tolower($3)]"\t"$4)}' broad_drug_info.txt drug_info | grep CTRP > drug_not_mapped.txt

awk -F'\t' 'NR==FNR {a[tolower($1)]=$NF;  a[tolower($2)]=$NF; next} {if((tolower($2) in a) || (tolower($3) in a)) print($1"\t"tolower($2)"\t"tolower($3)"\t"a[tolower($2)]"\t"a[tolower($3)]"\t"$4)}' broad_drug_info.txt drug_info | grep CTRP | tr -d '\r' > drug_mapped.txt

awk -F'\t' 'NR==FNR {a[$1];next} {if(!($1 in a)) print}' <(cat drug_mapped.txt drug_not_mapped.txt) drug_info  | grep CTRP > drug_left.txt

echo $'id\tsmiles' > cmpd_smiles.txt 
cat drug_mapped.txt drug_not_mapped.txt drug_left.txt | awk -F'\t' '{if($4=="") {print($1"\t"$NF)} else {print($1"\t"$4)}}' >> cmpd_smiles.txt


# Create AUC list
echo "broadid   cpdid   auc" > final_list_auc.txt
awk -F'\t' 'NR==FNR {a[$1];next} {if($2 in a)print($2"\t"$3"\t"$5"\t"$(NF-1))}' cells_with_gexp.txt combined_single_response_agg >> final_list_auc.txt