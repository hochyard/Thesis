import random
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

#handle covariate dataframe
cov_matrix = pd.read_csv("/sise/nadav-group/nadavrap-group/UKBB/phenotypes/Genotype_batch.txt",sep=" ")
old_cov = pd.read_csv("/storage/users/nadavrap-group/Yarden/phenotypes/covariance_matrix3",sep=" ")
cov_matrix = pd.merge(cov_matrix, old_cov[['FID','genetic_sex_f22001_0_0']], on='FID')
cov_matrix = cov_matrix.drop(['sex_f31_0_0','uk_biobank_assessment_centre_f54_1_0','uk_biobank_assessment_centre_f54_2_0','uk_biobank_assessment_centre_f54_3_0'],axis=1)
cov_matrix = cov_matrix.dropna(axis=0).reset_index(drop=True)


#train test split
df_test_FID = pd.read_csv('/home/hochyard/UKBB/results/data_for_model/test_ID.dat',header=None)
df_train_FID = pd.read_csv('/home/hochyard/UKBB/results/data_for_model/train_ID.dat',header=None)
cov_matrix['FID']=cov_matrix['FID'].astype(str)
train_cov_matrix = cov_matrix[cov_matrix['FID'].isin(df_train_FID.astype(str).values.flatten())].reset_index(drop=True)
test_cov_matrix = cov_matrix[cov_matrix['FID'].isin(df_test_FID.astype(str).values.flatten())].reset_index(drop=True)


enc = OneHotEncoder(handle_unknown='error', drop='if_binary')
train_cov_matrix_dummy = pd.DataFrame(enc.fit_transform(train_cov_matrix[['uk_biobank_assessment_centre_f54_0_0','genetic_sex_f22001_0_0']]).toarray(),columns=[enc.get_feature_names_out()])
train_cov_matrix = train_cov_matrix.drop(['uk_biobank_assessment_centre_f54_0_0','genetic_sex_f22001_0_0'], axis=1)
train_cov_matrix = pd.concat([train_cov_matrix, train_cov_matrix_dummy], axis = 1)

test_cov_matrix_dummy = pd.DataFrame(enc.transform(test_cov_matrix[['uk_biobank_assessment_centre_f54_0_0','genetic_sex_f22001_0_0']]).toarray(),columns=[enc.get_feature_names_out()])
test_cov_matrix = test_cov_matrix.drop(['uk_biobank_assessment_centre_f54_0_0','genetic_sex_f22001_0_0'], axis=1)
test_cov_matrix = pd.concat([test_cov_matrix, test_cov_matrix_dummy], axis = 1)

cov_dummy = pd.concat([train_cov_matrix, test_cov_matrix], axis = 0).reset_index(drop=True)
IID = cov_dummy['IID']
cov_dummy = cov_dummy.drop(['IID'], axis=1)
cov_dummy.to_pickle('/home/hochyard/UKBB/results/data_for_model/cov_matrix_no_missing_no_40PCA.pkl')
cov_dummy.insert(1, "IID", IID)
cov_dummy.to_csv('/home/hochyard/UKBB/results/data_for_model/cov_matrix_no_missing_no_40PCA.txt', header=True, index=False, sep='\t')

#----scale cov----
scaler_cov = MinMaxScaler()
train_scaled_values = pd.DataFrame(scaler_cov.fit_transform(train_cov_matrix[['age_when_attended_assessment_centre_f21003_0_0','genotype_measurement_batch_f22000_0_0']]), index = train_cov_matrix.index, columns = ['age_when_attended_assessment_centre_f21003_0_0','genotype_measurement_batch_f22000_0_0'])
train_cov_matrix_scaled = train_cov_matrix.drop(['age_when_attended_assessment_centre_f21003_0_0','genotype_measurement_batch_f22000_0_0'], axis=1)
train_cov_matrix_scaled = pd.concat([train_cov_matrix_scaled, train_scaled_values], axis = 1)

test_scaled_values = pd.DataFrame(scaler_cov.transform(test_cov_matrix[['age_when_attended_assessment_centre_f21003_0_0','genotype_measurement_batch_f22000_0_0']]), index = test_cov_matrix.index, columns = ['age_when_attended_assessment_centre_f21003_0_0','genotype_measurement_batch_f22000_0_0'])
test_cov_matrix_scaled = test_cov_matrix.drop(['age_when_attended_assessment_centre_f21003_0_0','genotype_measurement_batch_f22000_0_0'], axis=1)
test_cov_matrix_scaled = pd.concat([test_cov_matrix_scaled, test_scaled_values], axis = 1)

cov_dummy_scaled = pd.concat([train_cov_matrix_scaled, test_cov_matrix_scaled], axis = 0).reset_index(drop=True)
IID = cov_dummy_scaled['IID']
cov_dummy_scaled = cov_dummy_scaled.drop(['IID'], axis=1)
cov_dummy_scaled.to_pickle('/home/hochyard/UKBB/results/data_for_model/cov_matrix_MinMax_scaled_no_missing_no_40PCA.pkl')
cov_dummy_scaled.insert(1, "IID", IID)
cov_dummy_scaled.to_csv('/home/hochyard/UKBB/results/data_for_model/cov_matrix_MinMax_scaled_no_missing_no_40PCA.txt', header=True, index=False, sep='\t')

