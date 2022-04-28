import pandas as pd
from pandas_plink import read_plink
import random
import numpy as np

file_name = '/home/hochyard/UKBB/results/chr1_imputed_snp_id_filtered_no_missing.bed' #after removing "ids_to_delete"
#call file with Europeans' ID?????????????????????????
(bim, fam, bed) = read_plink(file_name, verbose=False)
ID = fam['fid'].to_frame()
ID = ID.rename(columns = {'fid': 'FID'}, inplace = False)
test_size = int(len(list(ID['FID']))*0.15)
test_FID = random.sample(list(ID['FID']), test_size)
df_test_FID = pd.DataFrame(test_FID,columns = ['FID']).sort_values('FID').reset_index(drop=True)
df_train_FID = ID[~ID['FID'].isin(test_FID)].sort_values('FID').reset_index(drop=True)

df_test_FID.to_csv('/home/hochyard/UKBB/results/data_for_model/test_ID.dat')
df_train_FID.to_csv('/home/hochyard/UKBB/results/data_for_model/train_ID.dat')

#what about the validation??????????????????