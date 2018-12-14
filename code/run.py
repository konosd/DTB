import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import tensorflow as tf
import os
import random
import keras
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
import re
from keras import optimizers
from keras import losses
from keras import regularizers
from keras.models import model_from_json
from keras.models import load_model





tf.set_random_seed(1); np.random.seed(1); random.seed(1)

input_dir = os.path.join(os.getcwd(),'data')

df = pd.read_csv(os.path.join(input_dir,'davis_data_python.csv'),index_col=0)

kinases = df['GeneName']
kinases = kinases.drop_duplicates()
drugs = df['PubchemId']
drugs = drugs.drop_duplicates()
print(kinases.head())
print(drugs.head())
print([kinases.shape, drugs.shape])

df.head()

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
		
		
		
def dtb_model(params,lr_value, windows_smiles, windows_seq):
    XDinput = keras.layers.Input(shape = (100,dict_smiles_len))
    XTinput = keras.layers.Input(shape = (1000, dict_prot_len))

    encode_smiles = keras.layers.Conv1D(filters = windows_smiles, kernel_size = params['size'], activation='relu', padding = 'valid', strides = 1)(XDinput)
    encode_smiles = keras.layers.Conv1D(filters = 2*windows_smiles, kernel_size = params['size'], activation = 'relu', padding = 'valid', strides = 1)(encode_smiles)
    encode_smiles = keras.layers.Conv1D(filters = 3*windows_smiles, kernel_size = params['size'], activation = 'relu', padding = 'valid', strides = 1)(encode_smiles)
    encode_smiles = keras.layers.GlobalMaxPooling1D()(encode_smiles)

    encode_protein = keras.layers.Conv1D(filters = windows_seq, kernel_size = params['size'], activation='relu', padding = 'valid', strides = 1)(XTinput)
    encode_protein = keras.layers.Conv1D(filters = 2*windows_seq, kernel_size = params['size'], activation = 'relu', padding = 'valid', strides = 1)(encode_protein)
    encode_protein = keras.layers.Conv1D(filters = 3*windows_seq, kernel_size = params['size'], activation = 'relu', padding = 'valid', strides = 1)(encode_protein)
    encode_protein = keras.layers.GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])

    fc1 = keras.layers.Dense(params['dense_size'],activation = 'relu',kernel_regularizer=regularizers.l2(params['l2reg']))(encode_interaction)
    fc2 = keras.layers.Dropout(0.1)(fc1)
    fc2 = keras.layers.Dense(params['dense_size'],activation = 'relu',kernel_regularizer=regularizers.l2(params['l2reg']))(fc2)
    fc2 = keras.layers.Dropout(0.1)(fc2)
    fc2 = keras.layers.Dense(params['dense_size2'],activation = 'relu',kernel_regularizer=regularizers.l2(params['l2reg']))(fc2)

    predictions = keras.layers.Dense(1, kernel_initializer='normal')(fc2)

    interactionModel = keras.Model(inputs=[XDinput, XTinput], outputs=[predictions])

    print(interactionModel.summary())
    adam = keras.optimizers.Adam(lr=lr_value, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    interactionModel.compile(
        optimizer= adam, 
        loss='mean_squared_error', 
        metrics=['mse']
    )
    return interactionModel


XD = []
XT = []

for i in range(df.shape[0]):
	drug = one_hot_smiles(df.Isosmiles.iloc[i])
	XD.append(drug)

XD = np.array(XD)

for i in range(df.shape[0]):
	target = one_hot_sequence(df.Sequence.iloc[i])
	XT.append(target)

XT = np.array(XT)

Y = np.log10(df.KD)-np.mean(np.log10(df.KD))
Y = Y.values


p = {'lr': 0.001,
     'nfilters': int(32),
     'size': int(8),
     'batch_size': int(128),
     'dense_size': int(1024),
     'dense_size2': 512,
     'dropout': 0.1,
     'l2reg': 0.00}


lr_value = 0.001
all_scores = []
windows_smiles = [32,48,64]
windows_seq = [32,48]
for i in range(len(windows_smiles)):
	for j in range(len(windows_seq)):
		interactionModel = dtb_model(p,lr_value,windows_smiles[i],windows_seq[j])
		tensorboard = keras.callbacks.TensorBoard(log_dir='/artifacts', histogram_freq=0,  
          write_graph=True, write_images=True)
		interactionModel.fit([XD,XT],Y, batch_size = 128 , epochs = 20, callbacks=[tensorboard]) 
		scores = interactionModel.evaluate([XD,XT], Y, verbose=0)
		all_scores.append(scores[1])

print(all_scores)


