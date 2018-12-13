import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import random
import keras
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pubchempy as pcp
import re

tf.set_random_seed(1); np.random.seed(1); random.seed(1)

input_dir = os.path.join(os.getcwd(),'data')

# Reading the files that Chris sent, from the Davis dataset
# Drug compounds
drugs = pd.read_table(os.path.join(input_dir,'davis_pubchem.txt'),
                      header=None, names=['PubchemId'])

# Target kinases
targets = pd.read_table(os.path.join(input_dir, 'davis_targets.txt'),
                        header = None, names = ['GeneName'])

# Wide format affinities
kd_values = pd.read_table(os.path.join(input_dir, 'davis_kd_values.txt'),
                          delim_whitespace=True, header = None, names = targets.GeneName)

# Adding drug IDs from Pubchem
kd_values.insert(column= 'Drug', value=drugs.PubchemId, loc = 0)

# Melt dataset to tidy data
kd_values = kd_values.melt(id_vars='Drug', value_name='KD',var_name='Target')

# Getting isomerical smiles from pubchemID
def get_isosmiles(drugs, aslist = True):
    my_pubchem_list = list(map(pcp.Compound.from_cid, drugs))
    isosmiles = [c.isomeric_smiles for c in my_pubchem_list]
    if aslist == False:
        return pd.DataFrame(isosmiles, columns = ['Isosmiles'])
    else:
        return isosmiles

drugs.insert(column = "Isosmiles", value = get_isosmiles(drugs.PubchemId,aslist = True) ,loc = 1)

# Mapping names from LINCS database to Uniprot ID
lincs_db = pd.read_csv(os.path.join(input_dir,'lincs_prots.csv'),usecols=['Name','UniProtID'])

df_targets = pd.merge(left = targets, right = lincs_db, left_on='GeneName', right_on='Name',how='left')

# Removing the duplicate GeneNames AND UniProtIDs
df_targets = df_targets.drop(columns=['Name'])

df_targets = df_targets.drop_duplicates(subset=['GeneName'])

# There are some NaN's in the dataset. This is because the Davis dataset for god knows what reason uses alternative names.
# We will tidy those up
lincs_db = pd.read_csv(os.path.join(input_dir,'lincs_prots.csv'),usecols=['Name','UniProtID','AlternativeNames'])

proteins_with_no_id = df_targets[pd.isnull(df_targets.UniProtID)]
print('There are {} missing sequences'.format(proteins_with_no_id.shape[0]))

pd.set_option('mode.chained_assignment', None)

for p in range(proteins_with_no_id.shape[0]):
    #print('Searching for protein {}'.format(proteins_with_no_id.GeneName.iloc[p]))
    c = 0
    cache = []
    for i in range(lincs_db.shape[0]):
        #print('Searching in entry {}'.format(i))
        if pd.isnull(lincs_db.AlternativeNames.iloc[i]) == False:
            hit = re.findall(pattern= proteins_with_no_id.GeneName.iloc[p], string= lincs_db.AlternativeNames.iloc[i])
            if len(hit) > 0:
                c += 1
                cache.append(i)
    if len(cache)>0:
        proteins_with_no_id.iloc[p,1] = lincs_db.UniProtID.iloc[cache[0]]

pd.set_option('mode.chained_assignment', 'warn')

proteins_with_no_id.head()
# Ended up with only 11 missing. Bye Bye 11
proteins_with_no_id = proteins_with_no_id.dropna()

for i in range(df_targets.shape[0]):
    for p in range(proteins_with_no_id.shape[0]):
        if df_targets.iloc[i,0] == proteins_with_no_id.iloc[p,0]:
            df_targets.iloc[i,1] = proteins_with_no_id.iloc[p,1]

df_targets = df_targets.dropna()

# Getting rid of some problematic proteins
df_targets = df_targets[df_targets.UniProtID != 'P0A5S4']
df_targets = df_targets.drop_duplicates(subset=['UniProtID'])

# Getting AA sequence from Uniprot ID
from urllib.request import urlopen


def get_aaseq(uniprots, aslist = True):
    url_init = r'https://www.uniprot.org/uniprot/'
    aa_seq = []
    for protein in uniprots:
        print(protein)
        url = url_init + protein + r'.fasta'
        response = urlopen(url)
        fasta = response.read().decode('utf-8', 'ignore')
        aa_seq.append(''.join(fasta.split('\n')[1:-1]))
    if aslist == True:
        return aa_seq
    else:
        return pd.DataFrame(aa_seq, columns = 'Sequence')

df_targets.insert(column = 'Sequence', value = get_aaseq(list(df_targets.UniProtID)), loc = 2)

print(drugs.head())
print(kd_values.head())

df = pd.merge(kd_values, drugs, right_on='PubchemId',left_on='Drug')
df = df.drop(columns = ['Drug'])

print(df.head())
print(df_targets.head())

df = pd.merge(df, df_targets, right_on='GeneName',left_on='Target', how = 'left')
df = df.drop(columns = ['Target'])
df = df.dropna()

df.head()
# Saving to .csv file
df.to_csv('data/davis_data_python.csv')
