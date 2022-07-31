import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import os
import pickle
from pandas_plink import read_plink
import pandas as pd
import numpy as np
import time

def PCA_transform_incremental(PCA_transformer, X_chunks_file, output_file):
    with open(output_file, 'wb') as save_file_handle:
        with open(X_chunks_file, 'rb') as file_handle:
            while True:
                try:
                    batch = pickle.load(file_handle)
                    ID = batch['FID']
                    batch = batch.drop(['FID'], axis=1)
                    batch_PCA = PCA_transformer.transform(batch)
                    batch_PCA_df = pd.DataFrame(batch_PCA)
                    batch_PCA_df['FID'] = ID
                    pickle.dump(batch_PCA_df, save_file_handle, protocol=4)
                except EOFError:
                    break

def learn_PCA_matrix(X_chunks_file):
    #learn on all samples from train (independant from phenotype)
    with open(X_chunks_file, 'rb') as file_handle:
        batch = pickle.load(file_handle)    
    full_data = pd.DataFrame(columns=batch.keys())
    with open(X_chunks_file, 'rb') as file_handle:
        while True:
            try:
                batch = pickle.load(file_handle)
                full_data = full_data.append(batch)
            except EOFError:
                break
    pca = PCA(n_components=int((len(full_data.keys())-1)*0.1))
    full_data = full_data.drop(['FID'], axis=1)
    principalComponents = pca.fit(full_data)
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    return pca
 



for chr in range(int(os.environ['from']),int(os.environ['to'])):
    start_time = time.time()
    print(chr)                                     
    X_train_chunks_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_" + str(chr) +"_X_train_1k_chunks_no_missing.pkl"
    train_output_file = "/home/hochyard/my_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/chr_" + str(chr) + ".pkl"
    PCA_transformer = learn_PCA_matrix(X_train_chunks_file)
    pca_name = "/home/hochyard/my_model/PCA/PCA_transformer_" + str(chr) + ".pkl"
    pickle.dump(PCA_transformer, open(pca_name, 'wb'), protocol=4)
    print('Time to learn PCA for chromosome ', str(chr),':')
    print("--- %s seconds for PCA fiting---" % (time.time() - start_time))
    
    #predict
    #-----train------
    PCA_transformer = pickle.load(open(pca_name,'rb'))
    PCA_transform_incremental(PCA_transformer, X_train_chunks_file, train_output_file)
    #-----test------
    X_test_chunks_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_" + str(chr) +"_X_test_1k_chunks_no_missing.pkl"
    test_output_file = "/home/hochyard/my_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/chr_" + str(chr) + ".pkl"
    PCA_transform_incremental(PCA_transformer, X_test_chunks_file, test_output_file)
    
    

        