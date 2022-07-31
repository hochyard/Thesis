import random
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#handle covariate dataframe
cov_matrix = pd.read_csv("/storage/users/nadavrap-group/Yarden/phenotypes/covariance_matrix3",sep=" ")
cov_matrix_dummies = pd.get_dummies(cov_matrix[['uk_biobank_assessment_centre_f54_0_0','genetic_sex_f22001_0_0']], drop_first=True)
cov_matrix = cov_matrix.drop(['uk_biobank_assessment_centre_f54_0_0','genetic_sex_f22001_0_0'], axis=1)
cov_matrix = pd.concat([cov_matrix, cov_matrix_dummies], axis = 1)
cov_matrix = cov_matrix.rename(columns = {'#FID': 'FID'}, inplace = False)
cov_matrix['FID']=cov_matrix['FID'].astype(str)
cov_matrix = cov_matrix.dropna(axis=0).reset_index(drop=True)
remove =[]
IID = cov_matrix['IID']
cov_matrix = cov_matrix.drop(['IID'], axis=1)
cov_matrix.to_pickle('/home/hochyard/UKBB/results/data_for_model/cov_matrix_no_missing.pkl')
cov_matrix.insert(1, "IID", IID)
print(cov_matrix)
cov_matrix.to_csv('/home/hochyard/UKBB/results/data_for_model/cov_matrix_no_missing.txt', header=True, index=False, sep='\t')

#----scale cov----
#scaler_cov = StandardScaler()
scaler_cov = MinMaxScaler()
scaler_cov.fit(cov_matrix[['age_when_attended_assessment_centre_f21003_0_0']])
scaled_values = scaler_cov.transform(cov_matrix[['age_when_attended_assessment_centre_f21003_0_0']])
scaled_values = pd.DataFrame(scaled_values, index = cov_matrix.index, columns = ['age_when_attended_assessment_centre_f21003_0_0'])
cov_matrix['age_when_attended_assessment_centre_f21003_0_0'] = scaled_values
cov_matrix = cov_matrix.drop(['IID'], axis=1)
cov_matrix.to_pickle('/home/hochyard/UKBB/results/data_for_model/cov_matrix_MinMax_scaled_no_missing.pkl')
cov_matrix.insert(1, "IID", IID)
cov_matrix.to_csv('/home/hochyard/UKBB/results/data_for_model/cov_matrix_MinMax_scaled_no_missing.txt', header=True, index=False, sep='\t')


    

