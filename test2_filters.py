import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import tensorflow as tf
import os
import random
import keras
import sklearn
from sklearn.model_selection import KFold
from sklearn import metrics
import re
from keras import optimizers
from keras import losses
from keras import regularizers
from keras.models import model_from_json
from keras.models import load_model
from tempfile import TemporaryFile

# Setting up for reproducibility
tf.set_random_seed(1); np.random.seed(1); random.seed(1)

input_dir = os.path.join(os.getcwd(),'data')

# Reading data from davis
df = pd.read_csv(os.path.join(input_dir,'davis_data_python.csv'),index_col=0)
df = df.sample(frac=1).reset_index(drop=True)

# kinases = df['GeneName']
# kinases = kinases.drop_duplicates()
# drugs = df['PubchemId']
# drugs = drugs.drop_duplicates()
# print(kinases.head())
# print(drugs.head())
# print([kinases.shape, drugs.shape])

# df.head()

dict_prot = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                "U": 19, "T": 20, "W": 21,
                "V": 22, "Y": 23, "X": 24,
                "Z": 25 }
dict_prot_len = len(dict_prot)

dict_smiles = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
dict_smiles_len = len(dict_smiles)

# #########
# Encoders: one-hot and label
# #########
def one_hot_smiles(drug, max_smiles_len = 100, smi_dict = dict_smiles):
    X = np.zeros((max_smiles_len, len(smi_dict)))
    for i,ch in enumerate(drug[:max_smiles_len]):
        X[i, (smi_dict[ch])-1] = 1
    return X

def one_hot_sequence(protein, max_prot_len = 1000, prot_dict = dict_prot):
    X = np.zeros((max_prot_len, len(prot_dict)))
    for i,ch in enumerate(protein[:max_prot_len]):
        X[i, (prot_dict[ch])-1] = 1
    return X

def label_smiles(drug, max_smiles_len = 100, smi_dict = dict_smiles):
    X = np.zeros(max_smiles_len)
    for i,ch in enumerate(drug[:max_smiles_len]):
        X[i] = smi_dict[ch]

def label_sequence(protein, max_prot_len = 1000, prot_dict = dict_prot):
    X = np.zeros(max_prot_len)
    for i,ch in enumerate(protein[:max_prot_len]):
        X[i] = prot_dict[ch]

# ##### CIndex
def get_cindex(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f)

# #########
# Building the model
# #########
def dtb_model(params,lr_value, windows_smiles, windows_seq):
    XDinput = keras.layers.Input(shape = (100,dict_smiles_len))
    XTinput = keras.layers.Input(shape = (1000, dict_prot_len))

    encode_smiles = keras.layers.Conv1D(filters = windows_smiles, kernel_size = params['size_drug_1'], activation= 'relu', padding = 'same', strides = 1)(XDinput)
    encode_smiles = keras.layers.Conv1D(filters = 2*windows_smiles, kernel_size = params['size_drug_1'], activation = 'relu', padding = 'same', strides = 1)(encode_smiles)
    encode_smiles = keras.layers.Conv1D(filters = 3*windows_smiles, kernel_size = params['size_drug_1'], activation = 'relu', padding = 'same', strides = 1)(encode_smiles)
    encode_smiles = keras.layers.GlobalMaxPooling1D()(encode_smiles)

    encode_protein = keras.layers.Conv1D(filters = windows_seq, kernel_size = params['size_protein_1'], activation= 'relu', padding = 'same', strides = 1)(XTinput)
    encode_protein = keras.layers.Conv1D(filters = 2*windows_seq, kernel_size = params['size_protein_1'], activation = 'relu', padding = 'same', strides = 1)(encode_protein)
    encode_protein = keras.layers.Conv1D(filters = 3*windows_seq, kernel_size = params['size_protein_1'], activation = 'relu', padding = 'same', strides = 1)(encode_protein)
    encode_protein = keras.layers.GlobalMaxPooling1D()(encode_protein)


    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])


    fc1 = keras.layers.Dense(params['dense_size'],activation = 'relu',kernel_regularizer=regularizers.l2(params['l2reg']))(encode_interaction)
    fc2 = keras.layers.Dropout(0.2)(fc1)
    fc2 = keras.layers.Dense(params['dense_size'],activation = 'relu',kernel_regularizer=regularizers.l2(params['l2reg']))(fc2)
    fc2 = keras.layers.Dropout(0.2)(fc2)
    fc3 = keras.layers.Dense(params['dense_size_2'],activation = 'relu',kernel_regularizer=regularizers.l2(params['l2reg']))(fc2)


    predictions = keras.layers.Dense(1, kernel_initializer='normal')(fc3)

    interactionModel = keras.Model(inputs=[XDinput, XTinput], outputs=[predictions])

    print(interactionModel.summary())
    adam = keras.optimizers.Adam(lr=lr_value, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    interactionModel.compile(
        optimizer= adam,
        loss='mean_squared_error',
        metrics=['mse', get_cindex]
    )
    return interactionModel

# Cross Validation - data driven

def k_cross_val(XD, XT, Y, fold_num, keras_model, param_dict ):
    perf_vector = []
    kf = KFold(n_splits = fold_num)
    lr_value = 0.001
    windows_smiles = [32]
    windows_seq = [32]
    for i in range(len(windows_smiles)):
        for j in range(len(windows_seq)):
            all_ci = []
            all_mse = []
            cv_ci = []
            cv_mse = []
            tensorboard = keras.callbacks.TensorBoard(log_dir='/artifacts', histogram_freq=0,
              write_graph=True, write_images=True)
            for train_index, test_index in kf.split(Y):
                interactionModel = keras_model(p,lr_value,windows_smiles[i],windows_seq[j])
                interactionModel.fit([XD[train_index],XT[train_index]],Y[train_index], batch_size = 128 , epochs = 40, callbacks= [tensorboard] ,validation_data = ([XD[test_index],XT[test_index]], Y[test_index]))
                scores = interactionModel.evaluate([XD[test_index],XT[test_index]], Y[test_index], verbose=0)
                all_mse.append(scores[1])
                all_ci.append(scores[2])
            cv_ci = np.mean(all_ci)
            cv_mse = np.mean(all_mse)
            perf_vector.append([windows_smiles[i],windows_seq[j],cv_mse,cv_ci])
    np.savetxt('/artifacts/performance.txt',perf_vector)
    print(perf_vector)






# Passing in the data as pairs of interactions
def make_datasets(dataframe):
    XD = []
    XT = []

    for i in range(dataframe.shape[0]):
        drug = one_hot_smiles(dataframe.Isosmiles.iloc[i])
        XD.append(drug)
    XD = np.array(XD)

    print('Drugs encoded and ready for input')
    for i in range(dataframe.shape[0]):
        target = one_hot_sequence(dataframe.Sequence.iloc[i])
        XT.append(target)
    XT = np.array(XT)
    print('Kinases encoded and ready for input')

    Y = np.log10(dataframe.KD)-np.mean(np.log10(dataframe.KD))
    Y = Y.values
    print('Affinities logged, normalized, and ready for input')
    return(XD,XT, Y)

# Dictionary of parameters
p = {'lr': 0.001,
     'nfilters': int(32),
     'size': int(8),
     'size_drug_1' : 8,
     'size_drug_2' : 4,
     'size_protein_1' : 8,
     'size_protein_2' : 16,
     'size_protein_3' : 3,
     'batch_size': int(128),
     'dense_size': int(1024),
     'dense_size_2': 512,
     'dropout': 0.15,
     'l2reg': 0}

# OLA ta apokatw ta evala mesa sto k_cross_val
# Grid model
# lr_value = 0.0003
# all_scores = []
# windows_smiles = [48]
# windows_seq = [48]
# for i in range(len(windows_smiles)):
# 	for j in range(len(windows_seq)):
# 		interactionModel = dtb_model(p,lr_value,windows_smiles[i],windows_seq[j])
# 		tensorboard = keras.callbacks.TensorBoard(log_dir='/artifacts', histogram_freq=0,
#           write_graph=True, write_images=True)
# 		interactionModel.fit([XD,XT],Y, batch_size = 256 , epochs = 40, callbacks=[tensorboard], validation_split = 0.1)
# 		scores = interactionModel.evaluate([XD,XT], Y, verbose=0)
# 		all_scores.append(scores[1])
#
# print(all_scores)

XD, XT, Y = make_datasets(df)

k_cross_val(XD, XT, Y, 10, dtb_model, p)

#dtb_model(p,0.01,256,256)
