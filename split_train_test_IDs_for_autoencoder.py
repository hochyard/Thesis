import pandas as pd
from pandas_plink import read_plink
import random
import numpy as np

file_name = '/home/hochyard/UKBB/results/chr1_imputed_snp_id_filtered_no_missing.bed' #after removing "ids_to_delete"
ethnicity_file =  '/storage/users/nadavrap-group/UKBB/phenotypes/Ethnicity.pheno'
ethnicity = pd.read_csv(ethnicity_file,sep=" ")
ethnicity = ethnicity.loc[ethnicity["genetic_ethnic_grouping_f22006_0_0"] == 'Caucasian']
ethnicity['FID']=ethnicity['FID'].astype(str)

(bim, fam, bed) = read_plink(file_name, verbose=False)
ID = fam['fid'].to_frame()
ID = ID.rename(columns = {'fid': 'FID'}, inplace = False)
ID = ID[ID['FID'].isin(ethnicity['FID'])].reset_index(drop=True)

test_size = int(len(list(ID['FID']))*0.15)
test_FID = random.sample(list(ID['FID']), test_size)
df_test_FID = pd.DataFrame(test_FID,columns = ['FID']).sort_values('FID').reset_index(drop=True)
df_train_FID = ID[~ID['FID'].isin(test_FID)].sort_values('FID').reset_index(drop=True)

df_test_FID.to_csv('/home/hochyard/UKBB/results/data_for_model/test_ID.dat', header=False, index=False)
df_train_FID.to_csv('/home/hochyard/UKBB/results/data_for_model/train_ID.dat', header=False, index=False)

