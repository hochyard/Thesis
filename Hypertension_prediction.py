import pickle
import xgboost
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
import matplotlib.pyplot as plt
import torch
from copy import deepcopy

def build_DNN_model(number_features, LR):
    """
    Define NN model
    :param number_features (int): number of features as input to the model
    :param LR (float): learning rate
    :return: model pre-trained
    """
    model = models.Sequential(name='DNN')
    model.add(layer=layers.Dense(units=number_features*0.2, activation=layers.PReLU(), input_shape=[number_features]))
    model.add(layers.Dropout(0.1))
    model.add(layer=layers.Dense(units=number_features*0.1, activation=layers.PReLU()))
    model.add(layers.Dropout(0.1))
    model.add(layer=layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=Adam(learning_rate=LR))
    model.summary()
    return model

def train_DNN_incremental(x_train_chunks_file, y_train_chunks_file, X_val, y_val, batch_size, num_epochs, LR, directory_to_save):
    """
    Train Deep NN.
    Note: the samples of x and y should be arranged in the same order according to the ID.
    :param x_train_chunks_file (str): file name with the train data saved in chunks.
    :param y_train_chunks_file (str): file name with the train target data saved in chunks.
    :param X_val (df): validation set
    :param y_val (df): validation target data
    :param batch_size: the size of the batch based on which the model is trained.
    :param num_epochs: number of epochs to train.
    :param LR: learning rate.
    :param directory_to_save: the directory to save in.
    :return: trained DNN model.
    """
    loss = []
    val_loss = []
    for epoch in range(1, num_epochs + 1):
        print('Epoch = ', epoch)
        batch_train_loss = []
        with open(x_train_chunks_file, 'rb') as x_file_handle:
            with open(y_train_chunks_file, 'rb') as y_file_handle:
                batch_num = 1
                while True:
                    try:
                        X_batch = pickle.load(x_file_handle).drop(['FID'], axis=1)
                        y_batch = pickle.load(y_file_handle).drop(['FID'], axis=1)
                        if epoch == 1 and batch_num == 1:
                            number_features = X_batch.shape[1]
                            model = build_DNN_model(number_features, LR)
                            best_model = build_DNN_model(number_features, LR)
                        if batch_num == 1: #its the validation set
                            batch_num = batch_num + 1
                            continue
                        history = model.fit(X_batch, y_batch, epochs=1, batch_size=batch_size, validation_data=(X_val, y_val))
                        batch_num = batch_num + 1
                        batch_train_loss.extend(history.history['loss'])
                    except EOFError:
                        break

        loss.append(np.average(batch_train_loss))
        val_loss.extend(history.history['val_loss'])

        #check early stopping - after 6
        if epoch == 1:
            best_result = history.history['val_loss']
            best_result_counter = 0
            continue
        if best_result > history.history['val_loss']:
            best_model.set_weights(model.get_weights())
            best_result = history.history['val_loss']
            best_result_counter = 0
        else:
            best_result_counter += 1

        if best_result_counter == 6:
            model.set_weights(best_model.get_weights())
            loss = loss[:-6]
            val_loss = val_loss[:-6]
            break

    save_name = directory_to_save + "loss_DNN.png"
    title = 'Autoencoder + DNN Loss on Hypertension'
    plot_loss(loss, val_loss, title, save_name)
    return model

def train_XGBClassifier_incremental(x_train_chunks_file, y_train_chunks_file, X_val, y_val, max_depth, num_epochs, LR, colsample_bytree, gamma, directory_to_save):
    """
    Train XGBClassifier. The model is learning incrementally, for each epoch, the model build by each chunk, one gradient boosted tree.
    Note: the samples of x and y should be arranged in the same order according to the ID.
    :param x_train_chunks_file (str): file name with the train data saved in chunks.
    :param y_train_chunks_file (str): file name with the train target data saved in chunks.
    :param X_val (df):  validation set
    :param y_val (df): validation target data
    :param max_depth: maximum depth of a tree.
    :param num_epochs: number of epochs to train.
    :param LR: learning rate.
    :param colsample_bytree: subsample ratio of columns when constructing each tree.
    :param gamma: minimum loss reduction required to make a further partition on a leaf node of the tree.
    :param directory_to_save: the directory to save the in.
    :return: trained XGBClassifier model.
    """
    loss = []
    val_loss = []
    model = xgboost.XGBClassifier(n_estimators=1, max_depth=max_depth, learning_rate=LR, colsample_bytree=colsample_bytree, gamma=gamma)
    #default objective=binary:logistic
    #default eval_metric=error
    for epoch in range(1,num_epochs+1):
        print('Epoch = ', epoch)
        batch_train_loss = []
        with open(x_train_chunks_file, 'rb') as x_file_handle:
            with open(y_train_chunks_file, 'rb') as y_file_handle:
                batch_num = 1
                while True:
                    try:
                        X_batch = pickle.load(x_file_handle).drop(['FID'], axis=1)
                        y_batch = pickle.load(y_file_handle).drop(['FID'], axis=1)
                        if batch_num == 1:  #its the validation set
                            batch_num = batch_num + 1
                            continue
                        if batch_num == 2 and epoch==1:
                            model.fit(X_batch, y_batch, eval_metric='logloss', eval_set=[(X_batch, y_batch), (X_val, y_val)], verbose=True)
                        else:
                            model.fit(X_batch, y_batch, xgb_model=model, eval_metric='logloss', eval_set=[(X_batch, y_batch), (X_val, y_val)], verbose=True)
                        batch_num = batch_num + 1
                        results = model.evals_result()
                        batch_train_loss.extend(results['validation_0']['logloss'])
                    except EOFError:
                        break
        loss.append(np.average(batch_train_loss))
        val_loss.extend(results['validation_1']['logloss'])

        # check early stopping - after 6
        if epoch == 1:
            best_result = results['validation_1']['logloss']
            best_result_counter = 0
            continue
        if best_result > results['validation_1']['logloss']:
            best_model = deepcopy(model)
            best_result = results['validation_1']['logloss']
            best_result_counter = 0
        else:
            best_result_counter += 1

        if best_result_counter == 6:
            model = deepcopy(best_model)
            loss = loss[:-6]
            val_loss = val_loss[:-6]
            break
    save_name = directory_to_save + "loss_XGB.png"
    title = 'Autoencoder + XGB Loss on Hypertension'
    plot_loss(loss, val_loss, title, save_name)
    return model

def plot_loss(loss, val_loss, title, save_name):
    """
    Plot DNN loss dependent on the number of epochs.
    :param loss (array): train loss for each epoch
    :param val_loss (array): validation loss for each epoch
    """
    fig = plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.xlim(0, len(loss))
    plt.ylim(0, max(np.max(loss),np.max(val_loss)))
    fig.savefig(save_name)

def create_validation_set(X_train_chunks_file, y_train_chunks_file):
    """
    create validation set - the first chunk from the input file.
    Note: the samples of x and y should be arranged in the same order according to the ID.
    :param X_train_chunks_file: file name with the train data saved in chunks.
    :param y_train_chunks_file: file name with the train target data saved in chunks.
    :return: two datasets. X_val - x validation
                           y_val - y validation
    """
    with open(X_train_chunks_file, 'rb') as x_file_handle:
        X_val = pickle.load(x_file_handle).drop(['FID'], axis=1)
    with open(y_train_chunks_file, 'rb') as y_file_handle:
        y_val = pd.read_pickle(y_file_handle).drop(['FID'], axis=1)
    return X_val, y_val

def predict_trait(model, x_chunks_file):
    """
    Predict the trait - target variable base on the data.
    :param model: the model on the basis of which the prediction is made
    :param x_chunks_file (str): file name with the data saved in chunks.
    :return (array): target predicted values for each sample.
    """
    predictions = []
    with open(x_chunks_file, 'rb') as x_file_handle:
        while True:
            try:
                X_batch = pickle.load(x_file_handle).drop(['FID'], axis=1)
                prediction = model.predict(X_batch)
                predictions.extend(prediction.flatten())
            except EOFError:
                break
    return predictions

def model_evaluation(model, x_chunks_file, y):
    """
    Eavluate the given model on the given dataset using RMSE and R squared.
    Note: the samples of x and y should be arranged in the same order according to the ID.
    :param model:
    :param x_chunks_file (str): file name with the data saved in chunks.
    :param y (array-like): target variables of shape (n_samples,)
    """
    predictions = predict_trait(model=model,
                                x_chunks_file=x_chunks_file)
    print('log_loss =', log_loss(y, predictions))
    print('roc_auc_score =', roc_auc_score(y_test, predictions))
    print('average_precision_score =', average_precision_score(y_test, predictions))


if __name__ == '__main__':
    X_train_chunks_dim_remove_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001_batch_size250/X_train_1k_chunks_dim_remove_no_missing_600_epochs/hypertension_all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" #for Autoencoder
    X_test_chunks_dim_remove_file = "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001_batch_size250/X_test_1k_chunks_dim_remove_no_missing_600_epochs/hypertension_all_chr_MinMax_scaled_cov_MinMax_scaled_no_40PCA.pkl" #for Autoencoder
    y_train_chunks_file = "/home/hochyard/UKBB/results/data_for_model/hypertension_y_train_1k_chunks_no_missing.pkl"
    y_test_file = "/home/hochyard/UKBB/results/data_for_model/hypertension_y_test_no_missing.pkl"
    y_test = pd.read_pickle(y_test_file).drop(['FID'], axis=1)

    # create validation set
    X_val, y_val = create_validation_set(X_train_chunks_file=X_train_chunks_dim_remove_file,
                                         y_train_chunks_file=y_train_chunks_file)
    # train NN on training set
    print('available GPUs: ')
    print(torch.cuda.device_count())

    DNN_params = {'batch_size': 50,
                  'num_epochs': 80,
                  'LR': 0.0000001,
                  'directory_to_save': "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001_batch_size250/hypertension_pheno/NN/NN_0.2_dropout_0.1_dropout_relu_Adam0.0000001_batch_size_50_genes_MinMax_scaled_cov_MinMax_scaled_no_40_PCA/"}

    DNN_model = train_DNN_incremental(x_train_chunks_file=X_train_chunks_dim_remove_file,
                                      y_train_chunks_file=y_train_chunks_file,
                                      X_val=X_val,
                                      y_val=y_val,
                                      **DNN_params)

    model_evaluation(model=DNN_model,
                     x_chunks_file=X_test_chunks_dim_remove_file,
                     y=y_test)

    # train XGBoost on train set
    XGB_params = {'max_depth': 2,
                  'num_epochs': 20,
                  'LR': 0.001,
                  'colsample_bytree': 0.2,
                  'gamma': 0.01,
                  'directory_to_save': "/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001_batch_size250/hypertension_pheno/XGB/incremental_xgboost_clasifiier_MinMax_scaled_cov_MinMax_scaled_max_depth5_learning_rate0.001_subsample1.0_olsample_bytree0.2_gamma0.01_no_40_PCA/"}

    XGB_model = train_XGBClassifier_incremental(x_train_chunks_file=X_train_chunks_dim_remove_file,
                                               y_train_chunks_file=y_train_chunks_file,
                                               X_val=X_val,
                                               y_val=y_val,
                                               **XGB_params)

    model_evaluation(model=XGB_model,
                     x_chunks_file=X_test_chunks_dim_remove_file,
                     y=y_test)