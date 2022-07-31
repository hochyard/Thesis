import numpy as np
from pandas_plink import read_plink
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping

import torch
from math import sqrt
from sklearn.metrics import r2_score,log_loss, roc_auc_score, roc_curve, average_precision_score
import pickle
import matplotlib.pyplot as plt


def build_model(batch):
    # Define model
    number_snps = batch.shape[1]
    model = models.Sequential(name='NN')
    model.add(layer=layers.Dense(units=number_snps*0.2, activation=layers.PReLU(), input_shape=[number_snps]))
    model.add(layers.Dropout(0.1))
    model.add(layer=layers.Dense(units=number_snps*0.1, activation=layers.PReLU()))
    model.add(layers.Dropout(0.1))
    model.add(layer=layers.Dense(units=1, activation='sigmoid'))

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=Adam(learning_rate=0.0000001))
    model.summary()
    return model

def fit_NN(x_train_chunks_file, y_train_chunks_file, num_epochs_from, num_epochs_to, X_val, y_val, start_time, model=None):
    loss = []
    val_loss = []
    for epoch in range(num_epochs_from, num_epochs_to + 1):
        print('Epoch = ', epoch)
        with open(x_train_chunks_file, 'rb') as file_handle:
            with open(y_train_chunks_file, 'rb') as file_handle2:
                batch_num = 1
                while True:
                    try:
                        X_batch = pickle.load(file_handle)
                        X_batch = X_batch.drop(['FID'], axis=1)
                        y_batch = pickle.load(file_handle2)
                        y_batch = y_batch.drop(['FID'], axis=1)
                        if epoch == 1 and batch_num == 1 and model==None:
                            model = build_model(X_batch)
                        if batch_num == 1: #its the validation set
                            batch_num = batch_num + 1
                            continue
                        # fit
                        es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, restore_best_weights=True)
                        history = model.fit(X_batch, y_batch, epochs=1, batch_size=50, validation_data=(X_val, y_val), callbacks=[es])
                        batch_num = batch_num + 1
                    except EOFError:
                        break
        if epoch%10 == 0:
            print('Time fiting', epoch,'epochs:')
            print('--- %s seconds---' % (time.time() - start_time))
            save_to = "/home/hochyard/my_model/PCA/hypertension_pheno/NN/no_40PCA/NN_0.2_dropout_0.1_dropout_prelu_Adam0.0000001_batch_size_50_genes_MinMax_scaled_cov_MinMax_scaled/"+str(epoch)
            model.save(save_to,num_epochs_to)
        loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
    plot_loss(loss, val_loss, num_epochs_to, num_epochs_from)
    return model
    
def plot_loss(loss, val_loss,num_epochs_to,num_epochs_from):
    fig = plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    temp = int(num_epochs_to-num_epochs_from + 5)
    plt.xlim(0, temp)
    plt.ylim(0, 1)
    name = "/home/hochyard/my_model/PCA/hypertension_pheno/NN/no_40PCA/NN_0.2_dropout_0.1_dropout_prelu_Adam0.0000001_batch_size_50_genes_MinMax_scaled_cov_MinMax_scaled/80_epochs.png"
    fig.savefig(name)
    
def predict_pheno(model, x_chunks_file):
    predictions = []
    with open(x_chunks_file, 'rb') as file_handle:
        while True:
            try:
                X_batch = pickle.load(file_handle)
                X_batch = X_batch.drop(['FID'], axis=1)
                prediction = model.predict(X_batch)
                predictions.extend(prediction.flatten())
            except EOFError:
                break
    return predictions

def create_validation_set(X_train_chunks_file, y_train_chunks_file):
    with open(X_train_chunks_file, 'rb') as file_handle:
        X_val = pickle.load(file_handle)
        X_val = X_val.drop(['FID'], axis=1)
    with open(y_train_chunks_file, 'rb') as y_file_handle:
        y_val = pd.read_pickle(y_file_handle)
        y_val = y_val.drop(['FID'], axis=1)
    return X_val, y_val
        
def plot_roc_curve(fpr, tpr):
    fig = plt.figure()
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    name = "/home/hochyard/my_model/PCA/hypertension_pheno/NN/no_40PCA/NN_0.2_dropout_0.1_dropout_prelu_Adam0.0000001_batch_size_50_genes_MinMax_scaled_cov_MinMax_scaled/80_epochs_ROC.png"
    fig.savefig(name)


   
import time
start_time = time.time()
print('available GPUs: ')
print(torch.cuda.device_count())

  
# files
X_train_chunks_dim_remove_file = "/home/hochyard/my_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/hypertension_all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" #for PCA
#X_train_chunks_dim_remove_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_train_1k_chunks_dim_remove_no_missing_1000_epochs/hypertension_all_chr_MinMax_scaled_cov_MinMax_scaled.pkl" #for Autoencoder

                     
X_test_chunks_dim_remove_file = "/home/hochyard/my_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/hypertension_all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" #for PCA
#X_test_chunks_dim_remove_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/X_test_1k_chunks_dim_remove_no_missing_1000_epochs/hypertension_all_chr_MinMax_scaled_cov_MinMax_scaled.pkl" #for Autoencoder

y_train_chunks_file = "/home/hochyard/UKBB/results/data_for_model/hypertension_y_train_1k_chunks_no_missing.pkl"
y_test_file = "/home/hochyard/UKBB/results/data_for_model/hypertension_y_test_no_missing.pkl"

# train NN on train set
val_size = 1000
X_val, y_val = create_validation_set(X_train_chunks_file = X_train_chunks_dim_remove_file,
                                     y_train_chunks_file = y_train_chunks_file)
model = fit_NN(x_train_chunks_file = X_train_chunks_dim_remove_file,
               y_train_chunks_file = y_train_chunks_file,
               num_epochs_from = 1,
               num_epochs_to = 80,
               X_val = X_val,
               y_val = y_val,
               start_time = start_time,
               model = None)

#-----------predict----------
#load NN model
#model_name = "/home/hochyard/my_model/PCA/hypertension_pheno/NN/NN_0.2_dropout_0.1_dropout_prelu_Adam0.0000001_batch_size_50_genes_MinMax_scaled_cov_MinMax_scaled/80"
#model = keras.models.load_model(model_name)

predictions = predict_pheno(model=model,
                            x_chunks_file = X_test_chunks_dim_remove_file)
predictions_df = pd.DataFrame(predictions,columns=['hypertension'])                            
predictions_df.to_csv("/home/hochyard/my_model/PCA/hypertension_pheno/NN/no_40PCA/NN_0.2_dropout_0.1_dropout_prelu_Adam0.0000001_batch_size_50_genes_MinMax_scaled_cov_MinMax_scaled/predictions_df_80_epochs")
#predictions_df = pd.read_csv("/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/hypertension_pheno/NN/NN_0.2_dropout_0.1_dropout_relu_Adam0.0000001_batch_size_50_genes_MinMax_scaled_cov_MinMax_scaled/predictions_df_80_epochs",header=0,index_col=0)
predictions_df = predictions_df.rename(columns={"0":"hypertension"})
predictions = predictions_df.to_numpy()
predictions_df=predictions_df.astype(float)
y_test = pd.read_pickle(y_test_file)
y_test = y_test.drop(['FID'], axis=1)
y_test=y_test.astype(int)
print('log_loss=', log_loss(y_test,predictions))
fpr, tpr, thresholds = roc_curve(y_test, predictions)
print('roc_auc_score=',roc_auc_score(y_test,predictions))
print('average_precision_score =',average_precision_score(y_test, predictions))
plot_roc_curve(fpr, tpr)

