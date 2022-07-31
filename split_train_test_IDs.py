import pandas as pd
from pandas_plink import read_plink
import random
import numpy as np

# handle phenotypes dataframe
pheno = pd.read_csv("/storage/users/nadavrap-group/Yarden/phenotypes/ukbb_physical_measures_filtered.txt", sep=" ")
pheno['FID']=pheno['FID'].astype(str)
pheno = pheno[['FID','standing_height_f50_0_0']]
pheno = pheno.dropna(axis=0).sort_values('FID').reset_index(drop=True)

#handle covariate dataframe
cov_matrix = pd.read_csv("/storage/users/nadavrap-group/Yarden/phenotypes/covariance_matrix3",sep=" ")
cov_matrix_dummies = pd.get_dummies(cov_matrix[['uk_biobank_assessment_centre_f54_0_0','genetic_sex_f22001_0_0']], drop_first=True)
cov_matrix = cov_matrix.drop(['uk_biobank_assessment_centre_f54_0_0','genetic_sex_f22001_0_0'], axis=1)
cov_matrix = pd.concat([cov_matrix, cov_matrix_dummies], axis = 1)
cov_matrix = cov_matrix.rename(columns = {'#FID': 'FID'}, inplace = False)
cov_matrix['FID']=cov_matrix['FID'].astype(str)
cov_matrix = cov_matrix.drop(['IID'], axis=1)
cov_matrix = cov_matrix.dropna(axis=0).reset_index(drop=True)


file_name = '/home/hochyard/UKBB/results/chr1_imputed_snp_id_filtered_no_missing.bed' #after removing "ids_to_delete"
(bim, fam, bed) = read_plink(file_name, verbose=False)
ID = fam['fid'].to_frame()
ID = ID.rename(columns = {'fid': 'FID'}, inplace = False)
ID = ID[ID['FID'].isin(pheno['FID'])]
ID = ID[ID['FID'].isin(cov_matrix['FID'])].reset_index(drop=True)
test_size = int(len(list(ID['FID']))*0.15)
test_FID = random.sample(list(ID['FID']), test_size)
df_test_FID = pd.DataFrame(test_FID,columns = ['FID']).sort_values('FID').reset_index(drop=True)
df_train_FID = ID[~ID['FID'].isin(test_FID)].sort_values('FID').reset_index(drop=True)
df_test_FID.to_csv('/home/hochyard/UKBB/results/data_for_model/height_test_ID.dat')
df_train_FID.to_csv('/home/hochyard/UKBB/results/data_for_model/height_train_ID.dat')

pheno = pheno[pheno['FID'].isin(ID['FID'])].sort_values('FID').reset_index(drop=True) #assuming all chr have the same samples ID
y_test = pheno[pheno['FID'].isin(test_FID)].reset_index(drop=True)
y_train = pheno[~pheno['FID'].isin(y_test['FID'])].reset_index(drop=True)
y_train.to_pickle('height_y_train.pkl')
y_test.to_pickle('height_y_test.pkl')

cov_matrix = cov_matrix[cov_matrix['FID'].isin(ID['FID'])].sort_values('FID').reset_index(drop=True) #assuming all chr have the same samples ID
cov_matrix_test = cov_matrix[cov_matrix['FID'].isin(test_FID].reset_index(drop=True)
cov_matrix_train = cov_matrix[~cov_matrix['FID'].isin(cov_matrix_test['FID'])].reset_index(drop=True)
cov_matrix_train.to_pickle('/home/hochyard/UKBB/results/data_for_model/height_cov_matrix_train.pkl')
cov_matrix_test.to_pickle('/home/hochyard/UKBB/results/data_for_model/height_cov_matrix_test.pkl')

