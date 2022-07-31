import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
    # Denoising Autoencoders
    encoder = models.Sequential(name='encoder')
    encoder.add(layer=layers.Dense(units=new_dim_snps*2, activation=layers.PReLU(), input_shape=[input_size]))
    encoder.add(layers.Dropout(0.1))
    encoder.add(layer=layers.Dense(units=new_dim_snps, activation=layers.PReLU()))
    
    decoder = models.Sequential(name='decoder')
    decoder.add(layer=layers.Dense(units=new_dim_snps*2, activation=layers.PReLU(), input_shape=[new_dim_snps]))
    decoder.add(layers.Dropout(0.1))
    decoder.add(layer=layers.Dense(units=input_size, activation=layers.PReLU()))
    
    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(loss="mse", optimizer=Adam(learning_rate=0.00001))
    autoencoder.summary()
    return autoencoder

def fit_autoencoder(num_epochs_from,num_epochs_to, X_train_chunks_file, X_val, start_time, autoencoder=None):
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
                    es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, restore_best_weights=True)
                    history = autoencoder.fit(batch, batch, epochs=1, batch_size=250, shuffle=True, validation_data=(X_val, X_val), callbacks=[es])
                    batch_num = batch_num + 1
                except EOFError:
                    break
        loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
        if epoch%100 == 0:
            print('Time fiting', epoch,'epoch:')
            print('--- %s seconds---' % (time.time() - start_time))
            save_to = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001_batch_size250/autoencoder_chr" + str(chr) + "_" + str(epoch) + "_epochs"
            autoencoder.save(save_to,num_epochs_to)
    plot_loss(loss, val_loss,num_epochs_to,num_epochs_from, chr)
    return autoencoder
    
def plot_loss(loss, val_loss,num_epochs_to,num_epochs_from, chr):
    fig = plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    title = 'Autoencoder Loss Chromosome ' + str(chr)
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    temp = int(num_epochs_to-num_epochs_from + 5)
    plt.xlim(0, temp)
    plt.ylim(0, 0.2)
    name = '/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001_batch_size250/' + str(chr) + 'loss_autoencoder_600epochs.png'
    fig.savefig(name)

def create_validation_set(X_train_chunks_file):
    with open(X_train_chunks_file, 'rb') as file_handle:
        X_val = pickle.load(file_handle)
        X_val = X_val.drop(['FID'], axis=1)
    return X_val

#------------------------------train---------------------------
import time
print('available GPUs: ')
print(torch.cuda.device_count())



#set variable 'from', 'to'
for chr in range(int(os.environ['from']),int(os.environ['to'])):
    start_time = time.time()
    print(chr)
    #load autoencoder 
    #auto_name = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/autoencoder_chr" + str(chr) + "_700_epochs"
    #autoencoder = keras.models.load_model(auto_name)                                                   
    X_train_chunks_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_" + str(chr) +"_X_train_1k_chunks_no_missing.pkl"
    val_size = 1000
    X_val = create_validation_set(X_train_chunks_file)
    autoencoder = fit_autoencoder(num_epochs_from=1,
                                  num_epochs_to=600,
                                  X_train_chunks_file=X_train_chunks_file,
                                  X_val=X_val,
                                  start_time = start_time,
                                  autoencoder = None)                                      
















