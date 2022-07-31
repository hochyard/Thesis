import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import pickle
from tensorflow.keras import Model
import numpy as np
import os
import pandas as pd

def predict_new_snps(autoencoder, X_chunks_file, output_file):
    encoder = autoencoder.get_layer('encoder')
    encoder.summary()
    with open(X_chunks_file, 'rb') as file_handle:
        with open(output_file, 'wb') as save_file_handle:
            while True:
                try:
                    batch = pickle.load(file_handle)
                    ID = batch['FID']
                    batch = batch.drop(['FID'], axis=1)
                    codings = encoder.predict(batch)
                    codings = pd.DataFrame(codings)
                    codings['FID'] = ID
                    pickle.dump(codings, save_file_handle)
                except EOFError:
                    break


for chr in range(int(os.environ['from']),int(os.environ['to'])):                    
    #load autoencoder 
    auto_name = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/autoencoder_chr" + str(chr) + "_1000_epochs"
    autoencoder = keras.models.load_model(auto_name)
    # predict new features for each batch
    X_train_chunks_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_" + str(chr) +"_X_train_1k_chunks_no_missing.pkl"
    X_test_chunks_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_" + str(chr) +"_X_test_1k_chunks_no_missing.pkl"
    train_output_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_train_1k_chunks_dim_remove_no_missing_1000_epochs/chr_" + str(chr) + ".pkl"
    predict_new_snps(autoencoder=autoencoder,
                      X_chunks_file=X_train_chunks_file,
                      output_file=train_output_file)
    # ------------------------------test-------------------------
    test_output_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_test_1k_chunks_dim_remove_no_missing_1000_epochs/chr_" + str(chr) + ".pkl"
    predict_new_snps(autoencoder=autoencoder,
                      X_chunks_file=X_test_chunks_file,
                      output_file=test_output_file)
        

