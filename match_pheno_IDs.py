import pandas as pd
import numpy as np
import pickle

def match(x_file_name, pheno_df, x_output_file):   
    with open(x_output_file, 'wb') as x_output_file_handle:
        with open(x_file_name, 'rb') as x_file_handle:
            while True:
                try:
                    X_batch = pickle.load(x_file_handle)
                    X_batch = X_batch[X_batch['FID'].isin(pheno_df['FID'])].reset_index(drop=True)
                    pickle.dump(X_batch, x_output_file_handle, protocol=4)
                except EOFError:
                    break
                
def split_y_train_to_chunks(y_output_file, pheno_df, x_chunk_file, pheno_name):
    with open(y_output_file, 'wb') as y_file_handle:
       with open(x_chunk_file, 'rb') as x_file_handle:
           while True:
               try:
                   X_batch = pickle.load(x_file_handle)
                   df = pd.merge(X_batch, pheno_df, on='FID') #in order to order samples in the same order as in genes
                   y_batch = df[['FID',pheno_name]].reset_index(drop=True)
                   pickle.dump(y_batch, y_file_handle, protocol=4) 
               except EOFError:
                   break
                               
def split_test_y_to_chunks(input_df, x_chunk_file, pheno_name):
    pheno_df = pd.DataFrame(columns=['FID',pheno_name])
    with open(x_chunk_file, 'rb') as x_file_handle:
        while True:
            try:
                X_batch = pickle.load(x_file_handle)
                df = pd.merge(X_batch, input_df, on='FID') #in order to order samples in the same order as in genes
                y_batch = df[['FID',pheno_name]].reset_index(drop=True)
                pheno_df = pheno_df.append(y_batch)
            except EOFError:
                return pheno_df

# handle phenotypes dataframe
pheno = pd.read_csv("/sise/nadav-group/nadavrap-group/UKBB/phenotypes/hypertension_0_1", sep=" ") #hypertension
#pheno = pd.read_csv("/storage/users/nadavrap-group/Yarden/phenotypes/ukbb_physical_measures_filtered.txt", sep=" ") #height
pheno['FID']=pheno['FID'].astype(str)
#pheno = pheno[['FID','standing_height_f50_0_0']]
pheno = pheno[['FID','hypertension']]
pheno = pheno.dropna(axis=0).reset_index(drop=True)


#train
#union_train_gene = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_train_1k_chunks_dim_remove_no_missing_1000_epochs/all_chr_MinMax_scaled_cov_MinMax_scaled.pkl" #for Autoencoder
union_train_gene = "/home/hochyard/my_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" #for PCA
#x_train_output_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_train_1k_chunks_dim_remove_no_missing_1000_epochs/hypertension_all_chr_MinMax_scaled_cov_MinMax_scaled.pkl" #for Autoencoder
x_train_output_file = "/home/hochyard/my_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/hypertension_all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" #for PCA

match(union_train_gene, pheno, x_train_output_file)
y_train_output_file = "/home/hochyard/UKBB/results/data_for_model/hypertension_y_train_1k_chunks_no_missing.pkl"
split_y_train_to_chunks(y_train_output_file, pheno, x_train_output_file, 'hypertension')


#test
#union_test_gene = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_test_1k_chunks_dim_remove_no_missing_1000_epochs/all_chr_MinMax_scaled_cov_MinMax_scaled.pkl" #for Autoencoder
union_test_gene = "/home/hochyard/my_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" #for PCA
#x_test_output_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_test_1k_chunks_dim_remove_no_missing_1000_epochs/hypertension_all_chr_MinMax_scaled_cov_MinMax_scaled.pkl" #for Autoencoder
x_test_output_file = "/home/hochyard/my_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/hypertension_all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" #for PCA

match(union_test_gene, pheno, x_test_output_file)
test_pheno_df = split_test_y_to_chunks(pheno, x_test_output_file, 'hypertension')
test_pheno_df = test_pheno_df.reset_index(drop=True)
test_pheno_df.to_pickle("/home/hochyard/UKBB/results/data_for_model/hypertension_y_test_no_missing.pkl")
