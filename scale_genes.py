import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

            
def StandardScaler_incremental(train_file_name_to_standard_scale, test_file_name_to_standard_scale):
    scaler = {}
    for chr in range(1, 23):
        dim_remove_file = train_file_name_to_standard_scale + str(chr) + ".pkl"
        with open(dim_remove_file, 'rb') as file_handle:
            #chr_scaler = StandardScaler()
            chr_scaler = MinMaxScaler()
            while True:
                try:
                    batch = pickle.load(file_handle)
                    batch = batch.drop(['FID'], axis=1)
                    chr_scaler.partial_fit(batch)
                except EOFError:
                    break
            scaler[chr] = chr_scaler
    scale_all_genes(train_file_name_to_standard_scale, scaler)
    scale_all_genes(test_file_name_to_standard_scale, scaler)
  
def scale_all_genes(file_name_to_standard_scale, scaler):
    for chr in range(1, 23):
        file_name = file_name_to_standard_scale + str(chr) + ".pkl"
        with open(file_name, 'rb') as file_handle:
            #output_file_name = file_name_to_standard_scale + str(chr) + "_scaled.pkl"
            output_file_name = file_name_to_standard_scale + str(chr) + "_MinMax_scaled.pkl"
            with open(output_file_name, 'wb') as gene_output_file_handle:
                while True:
                    try:
                        batch = pickle.load(file_handle)
                        ID = batch['FID']
                        batch = batch.drop(['FID'], axis=1)
                        batch_scaled = scaler[chr].transform(batch)
                        batch = pd.DataFrame(batch_scaled)
                        batch['FID'] = ID
                        pickle.dump(batch, gene_output_file_handle, protocol=4)
                    except EOFError:
                        break
                    
#scale and save genes
train_X_file_name = "/home/hochyard/my_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/chr_" #for PCA
#train_X_file_name = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_train_1k_chunks_dim_remove_no_missing_1000_epochs/chr_" #for Autoencoder

#test
test_X_file_name = "/home/hochyard/my_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/chr_" #for PCA
#test_X_file_name = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_test_1k_chunks_dim_remove_no_missing_1000_epochs/chr_" #for Autoencoder
StandardScaler_incremental(train_X_file_name, test_X_file_name)

