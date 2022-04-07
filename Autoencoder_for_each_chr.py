import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from keras.models import Sequential
#from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import pickle
from sklearn.metrics import mean_squared_error
import random
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision
import numpy as np
import sys
from pympler.asizeof import asizeof
import torch


def build_autoencoder(input_size):
    new_dim_snps = int(input_size * 0.1)
    #encoder = keras.Sequential([layers.Dense(new_dim_snps, input_shape=[input_size])])
    #Denoising Autoencoders
    encoder = keras.Sequential([layers.Dense(new_dim_snps, input_shape=[input_size], activation=layers.PReLU(), activity_regularizer=regularizers.l1(10e-5))])
    decoder = keras.Sequential([layers.Dense(input_size, input_shape=[new_dim_snps], activation=layers.PReLU())])
    autoencoder = keras.Sequential([encoder, decoder])
    autoencoder.compile(loss="mse", optimizer=SGD(learning_rate=0.01))
    autoencoder.summary()
    return autoencoder, encoder

def fit_autoencoder(num_epochs, X_train_chunks_file, X_val, autoencoder=None, encoder=None):
    for epoch in range(1, num_epochs + 1):
        print('Epoch = ', epoch)
        with open(X_train_chunks_file, 'rb') as file_handle:
            batch_num = 1
            while True:
                try:
                    batch = pickle.load(file_handle)
                    if epoch == 1 and batch_num == 1 and autoencoder==None:
                        number_snps = batch.shape[1]-1 #minus FID column
                        autoencoder , encoder = build_autoencoder(number_snps)
                    if batch_num == 1: #its the validation set
                        batch_num = batch_num + 1
                        continue
                    batch = batch.drop(['FID'], axis=1)
                    # fit
                    autoencoder.fit(batch, batch, epochs=1, batch_size=200, shuffle=True, validation_data=(X_val, X_val))
                    batch_num = batch_num + 1
                except EOFError:
                    break
        if epoch%100 == 0:
            save_to = "/home/hochyard/autoencoder_models_relu_act_no_cov/autoencoder_model_chr" + str(chr) + "_no_missing_" + str(epoch) + "_epochs_activation_relu_no_cov"
            autoencoder.save(save_to)
            save_to = "/home/hochyard/autoencoder_models_relu_act_no_cov/encoder_chr" + str(chr) + "_no_missing_" + str(epoch) + "_epochs_activation_relu_no_cov"
            encoder.save(save_to)
    return autoencoder, encoder

def predict_new_snps(encoder, X_chunks_file, output_file):
    with open(X_chunks_file, 'rb') as file_handle:
        with open(output_file, 'wb') as save_file_handle:
            while True:
                try:
                    batch = pickle.load(file_handle)
                    batch = batch.drop(['FID'], axis=1)
                    codings = encoder.predict(batch)
                    pickle.dump(codings, save_file_handle)
                except EOFError:
                    break

def create_validation_set(X_train_chunks_file):
    with open(X_train_chunks_file, 'rb') as file_handle:
        X_val = pickle.load(file_handle)
        X_val = X_val.drop(['FID'], axis=1)
    return X_val

#------------------------------train---------------------------
import time
start_time = time.time()

# files
for chr in range(19,20):
    print(chr)
    #load autoencoder
    #auto_name = "autoencoder_model_chr" + str(chr) + "_no_missing_300_epochs"
    #autoencoder = keras.models.load_model(auto_name)
    #encoder_name = "encoder_chr" + str(chr) + "_no_missing_300_epochs"
    #encoder = keras.models.load_model(encoder_name)                                                    
    X_train_chunks_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_" + str(chr) +"_height_X_train_1k_chunks_no_missing.pkl"
    X_test_chunks_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_" + str(chr) + "_height_X_test_1k_chunks_no_missing.pkl"
    val_size = 1000
    X_val = create_validation_set(X_train_chunks_file)
    autoencoder, encoder = fit_autoencoder(num_epochs=700,
                                            X_train_chunks_file=X_train_chunks_file,
                                            X_val=X_val,
                                            autoencoder = None,
                                            encoder = None)
    #save_to = "autoencoder_model_chr" + str(chr) + "_no_missing_400_epochs"
    #autoencoder.save(save_to)
    #save_to = "encoder_chr" + str(chr) + "_no_missing_400_epochs"
    #encoder.save(save_to)
    print("--- %s seconds for data fiting---" % (time.time() - start_time))

    ## predict new features for each batch
    #train_output_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_" + str(chr) + "_height_X_train_1k_chunks_dim_remove_no_missing_400_epochs.pkl"
    #predict_new_snps(encoder=encoder,
    #                  X_chunks_file=X_train_chunks_file,
    #                  output_file=train_output_file)
# ------------------------------test-------------------------
    #test_output_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_" + str(chr) + "_height_X_test_1k_chunks_dim_remove_no_missing_400_epochs.pkl"
    #predict_new_snps(encoder=encoder, X_chunks_file=X_test_chunks_file, output_file=test_output_file)














