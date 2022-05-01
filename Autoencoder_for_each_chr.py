import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from keras.models import Sequential
#from keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from sklearn.metrics import mean_squared_error
import random
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision
import numpy as np
import sys
from pympler.asizeof import asizeof
import torch
import matplotlib.pyplot as plt
import os

def build_autoencoder(input_size):
    new_dim_snps = int(input_size * 0.1)
    #encoder = keras.Sequential([layers.Dense(new_dim_snps, input_shape=[input_size])])
    #Denoising Autoencoders
    # model
    encoder = models.Sequential(name='encoder')
    encoder.add(layer=layers.Dense(units=new_dim_snps*2, activation=layers.PReLU(), input_shape=[input_size]))
    encoder.add(layers.Dropout(0.1))
    encoder.add(layer=layers.Dense(units=new_dim_snps, activation=layers.PReLU()))
    
    decoder = models.Sequential(name='decoder')
    decoder.add(layer=layers.Dense(units=new_dim_snps*2, activation=layers.PReLU(), input_shape=[new_dim_snps]))
    decoder.add(layers.Dropout(0.1))
    decoder.add(layer=layers.Dense(units=input_size, activation=layers.PReLU()))
    
    autoencoder = models.Sequential([encoder, decoder])

    #encoder = keras.Sequential([layers.Dense(new_dim_snps, input_shape=[input_size], activation=layers.PReLU(), activity_regularizer=regularizers.l1(10e-5))])
    #decoder = keras.Sequential([layers.Dense(input_size, input_shape=[new_dim_snps], activation=layers.PReLU())])
    #autoencoder = keras.Sequential([encoder, decoder])
    #autoencoder.compile(loss="mse", optimizer=SGD(learning_rate=0.01,momentum=0.1))
    autoencoder.compile(loss="mse", optimizer=Adam(learning_rate=0.00001))
    autoencoder.summary()
    return autoencoder

def fit_autoencoder(num_epochs_from,num_epochs_to, X_train_chunks_file, X_val, autoencoder=None):
    loss = []
    val_loss = []
    for epoch in range(num_epochs_from, num_epochs_to + 1):
        print('Epoch = ', epoch)
        with open(X_train_chunks_file, 'rb') as file_handle:
            batch_num = 1
            while True:
                try:
                    batch = pickle.load(file_handle)
                    if epoch == 1 and batch_num == 1 and autoencoder==None:
                        number_snps = batch.shape[1]-1 #minus FID column
                        autoencoder = build_autoencoder(number_snps)
                    if batch_num == 1: #its the validation set
                        batch_num = batch_num + 1
                        continue
                    batch = batch.drop(['FID'], axis=1)
                    # fit
                    es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, restore_best_weights=True)
                    history = autoencoder.fit(batch, batch, epochs=1, batch_size=250, shuffle=True, validation_data=(X_val, X_val), callbacks=[es])
                    batch_num = batch_num + 1
                except EOFError:
                    break
        loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
        if epoch%100 == 0:
            save_to = "/home/hochyard/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/autoencoder_chr" + str(chr) + "_" + str(epoch) + "_epochs"
            autoencoder.save(save_to,num_epochs_to)
            #save_to = "/home/hochyard/autoencoder_models_5_layers_prelu_act_no_cov/encoder_chr" + str(chr) + "_" + str(epoch) + "_epochs"
            #encoder.save(save_to)
    plot_loss(loss, val_loss)
    return autoencoder
    
def plot_loss(loss, val_loss):
    fig = plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.xlim(0, 305)
    plt.ylim(0, 0.15)
    name = '/home/hochyard/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/' + str(chr) + 'loss_autoencoder.png'
    fig.savefig(name)

#def predict_new_snps(encoder, X_chunks_file, output_file):
#    with open(X_chunks_file, 'rb') as file_handle:
#        with open(output_file, 'wb') as save_file_handle:
#            while True:
#                try:
#                    batch = pickle.load(file_handle)
#                    batch = batch.drop(['FID'], axis=1)
#                    codings = encoder.predict(batch)
#                    pickle.dump(codings, save_file_handle)
#                except EOFError:
#                    break

def create_validation_set(X_train_chunks_file):
    with open(X_train_chunks_file, 'rb') as file_handle:
        X_val = pickle.load(file_handle)
        X_val = X_val.drop(['FID'], axis=1)
    return X_val

#------------------------------train---------------------------
import time
start_time = time.time()
print('available GPUs: ')
print(torch.cuda.device_count())

# files
#set variable 'from', 'to'
for chr in range(os.environ['from'],os.environ['to']):
    print(chr)
    #load autoencoder 
    #auto_name = "/home/hochyard/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/autoencoder_chr" + str(chr) + "_200_epochs"
    #autoencoder = keras.models.load_model(auto_name)
    #encoder_name = "encoder_chr" + str(chr) + "_no_missing_300_epochs"
    #encoder = keras.models.load_model(encoder_name)                                                    
    X_train_chunks_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_" + str(chr) +"_X_train_1k_chunks_no_missing.pkl"
    X_test_chunks_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_" + str(chr) + "_X_test_1k_chunks_no_missing.pkl"
    val_size = 1000
    X_val = create_validation_set(X_train_chunks_file)
    autoencoder = fit_autoencoder(num_epochs_from=1,
                                  num_epochs_to=8,
                                  X_train_chunks_file=X_train_chunks_file,
                                  X_val=X_val,
                                  autoencoder = None)                                      
    print("--- %s seconds for data fiting---" % (time.time() - start_time))
















