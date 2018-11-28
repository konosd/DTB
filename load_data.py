import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Aminoacids Dictionary
aminoacids = np.array(['A', 'C', 'D', 'E', 'F',
					'G', 'H', 'I', 'K', 'L',
					'M', 'N', 'P','Q', 'R',
					'S', 'T', 'V', 'W', 'Y']).reshape(-1,1)
 
# Smiles Dictionary
smiles_dic = np.array([' ',
                  '#', '%', '(', ')', '+', '-', '.', '/',
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  '=', '@',
                  'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                  'R', 'S', 'T', 'V', 'X', 'Z',
                  '[', '\\', ']',
                  'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                  't', 'u']).reshape(-1,1)


def one_hot_target(sequence, dictionary = aminoacids):
	# Check and convert string to list of letters
	if type(sequence) is str:
		sequence = np.array(list(sequence)).reshape(-1,1)
	else:
		sequence = np.array(sequence).reshape(-1,1)

	one_hot_enc = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
	one_hot_enc.fit(dictionary)
	return( one_hot_enc.transform(sequence))
	
def one_hot_smiles(sequence, dictionary = smiles_dic):
# Check and convert string to list of letters
	if type(sequence) is str:
		sequence = np.array(list(sequence)).reshape(-1,1)
	else:
		sequence = np.array(sequence).reshape(-1,1)

	one_hot_enc = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
	one_hot_enc.fit(dictionary)
	return( one_hot_enc.transform(sequence))
	
# Load Data
davis = pd.read_csv('davis_data.csv', header = 0)
davis = davis.fillna(value = 10000)



