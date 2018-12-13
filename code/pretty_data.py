import gevent.monkey
gevent.monkey.patch_all()

import numpy as np
from chembl_webresource_client import *
import os
import pandas as pd
# #####################
#
#	Load data from DTC
#
# #####################

# Reading the DTC and keeping only relevant information
data_dir = os.path.join(os.getcwd(),'data')
dtc_path = os.path.join(data_dir, 'DTC_data.csv')

def dtc_parser(file = dtc_path):
	dtc_db = pd.read_csv(file, usecols = ['compound_id','target_id',
		'standard_type','standard_relation', 'standard_value', 'standard_units'])
	dtc_db = dtc_db.dropna()
	dtc_db = dtc_db[dtc_db.standard_type == 'KD']
	return(dtc_db)

dtc_db = dtc_parser(dtc_path)

# Tidy data
# Take care of uneven relations
row_index = dtc_db[dtc_db.standard_relation == '>'].index
column_index = ['standard_value']
dtc_db.loc[row_index, column_index] = 10000
dtc_db.loc[row_index, ['standard_relation']] = '='

row_index = dtc_db[dtc_db.standard_relation == '>='].index
dtc_db.loc[row_index, ['standard_relation']] = '='

row_index = dtc_db[dtc_db.standard_relation == '<='].index
dtc_db.loc[row_index, ['standard_relation']] = '='

row_index = dtc_db[dtc_db.standard_relation == '<'].index
dtc_db.loc[row_index, ['standard_relation']] = '='

row_index = dtc_db[dtc_db.standard_relation == '~'].index
dtc_db.loc[row_index, ['standard_relation']] = '='


# Take care of weird concentration units
dtc_db = dtc_db[dtc_db.standard_units == 'NM']

# Now only keep vital information
dtc_db = dtc_db.loc[dtc_db.index, ['compound_id','target_id','standard_value']]

print('DTC database cleaned and ready, with {} compounds'.format(dtc_db.shape[0]))
print(dtc_db.head())
print('Ready to query compound isomerical SMILES and kinases aminoacid sequences')

# creating resource object
