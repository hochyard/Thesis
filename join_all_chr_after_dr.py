import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def join(X_file_name, chunk_size, gene_output_file, cov_df, gene_scale):
    number_samples = 0 
    with open(gene_output_file, 'wb') as gene_output_file_handle:
        this_chunk = 1
        while True:
            try:
                for chr in range(1, 23):
                    dim_remove_file = X_file_name + str(chr) 
                    #dim_remove_file = dim_remove_file + "_scaled.pkl" if gene_scale else dim_remove_file + ".pkl"
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
cov_df = pd.read_pickle('/home/hochyard/UKBB/results/data_for_model/cov_matrix_MinMax_scaled_no_missing.pkl')
remove =[]
for i in range(1,41):
    genetic_str = 'genetic_principal_components_f22009_0_' + str(i)
    remove += [genetic_str]
cov_df = cov_df.drop(remove, axis=1)
chunk_size=1000
gene_scale = True
train_X_file_name = "/home/hochyard/my_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/chr_" #for PCA
#train_X_file_name = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_train_1k_chunks_dim_remove_no_missing_1000_epochs/chr_" #for autoencoder

test_X_file_name = "/home/hochyard/my_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/chr_" #for PCA
#test_X_file_name = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_test_1k_chunks_dim_remove_no_missing_1000_epochs/chr_" #for Autoencoder

#train
union_train_gene_output_file = "/home/hochyard/my_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" #for PCA
#union_train_gene_output_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_train_1k_chunks_dim_remove_no_missing_1000_epochs/all_chr_MinMax_scaled_cov_MinMax_scaled.pkl" #for Autoencoder
join(train_X_file_name, chunk_size, union_train_gene_output_file, cov_df, gene_scale)

#test
union_test_gene_output_file = "/home/hochyard/my_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" #for PCA
#union_test_gene_output_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_test_1k_chunks_dim_remove_no_missing_1000_epochs/all_chr_MinMax_scaled_cov_MinMax_scaled.pkl" #for Autoencoder
join(test_X_file_name, chunk_size, union_test_gene_output_file, cov_df, gene_scale)
