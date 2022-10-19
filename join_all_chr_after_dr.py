import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def join(X_file_name, chunk_size, gene_output_file, cov_df, gene_scale):
    with open(gene_output_file, 'wb') as gene_output_file_handle:
        this_chunk = 1
        while True:
            try:
                for chr in range(1, 23):
                    dim_remove_file = X_file_name + str(chr) 
                    dim_remove_file = dim_remove_file + "_MinMax_scaled.pkl" if gene_scale else dim_remove_file + ".pkl"
                    with open(dim_remove_file, 'rb') as dim_remove_file_handle:
                        for c in range(1,this_chunk+1):
                            batch = pickle.load(dim_remove_file_handle)
                        ID = batch['FID']
                        batch = batch.drop(['FID'], axis=1)
                        suffix = '_chr'+str(chr)
                        batch = batch.add_suffix(suffix)
                        batch['FID'] = ID
                        if chr == 1:
                            df = pd.merge(cov_df, batch, on='FID')
                        else:
                            df = pd.merge(df, batch, on='FID')
                pickle.dump(df, gene_output_file_handle, protocol=4)
                this_chunk = this_chunk + 1
            except EOFError:
                break

# handle covariate
cov_df = pd.read_pickle('/home/hochyard/UKBB/results/data_for_model/cov_matrix_MinMax_scaled_no_missing_no_40PCA.pkl')
chunk_size=1000
gene_scale = True

#------------autoencoder---------------
autoencoder_train_X_file_name = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001_batch_size250/X_train_1k_chunks_dim_remove_no_missing_600_epochs/chr_" 
autoencoder_test_X_file_name = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001_batch_size250/X_test_1k_chunks_dim_remove_no_missing_600_epochs/chr_" 
#train
autoencoder_union_train_gene_output_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001_batch_size250/X_train_1k_chunks_dim_remove_no_missing_600_epochs/all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl"
join(autoencoder_train_X_file_name, chunk_size, autoencoder_union_train_gene_output_file, cov_df, gene_scale)
#test
autoencoder_union_test_gene_output_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001_batch_size250/X_test_1k_chunks_dim_remove_no_missing_600_epochs/all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" 
join(autoencoder_test_X_file_name, chunk_size, autoencoder_union_test_gene_output_file, cov_df, gene_scale)

#------------PCA---------------
PCA_train_X_file_name = "/home/hochyard/my_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/chr_" 
PCA_test_X_file_name = "/home/hochyard/my_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/chr_" 
#train
PCA_union_train_gene_output_file = "/home/hochyard/my_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" 
join(PCA_train_X_file_name, chunk_size, PCA_union_train_gene_output_file, cov_df, gene_scale)
#test
PCA_union_test_gene_output_file = "/home/hochyard/my_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" 
join(PCA_test_X_file_name, chunk_size, PCA_union_test_gene_output_file, cov_df, gene_scale)
