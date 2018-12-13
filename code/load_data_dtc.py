# Load data for DTC (parsing smiles and aminoacids)
import gevent.monkey
gevent.monkey.patch_all()

import numpy as np
import pandas as pd
import os
from urllib.request import urlopen
from chembl_webresource_client.new_client import new_client

# Setup directories
data_dir = os.path.join(os.getcwd(),'data')

# Import DTC database
dtc_db = pd.read_csv(os.path.join(data_dir,'dtc_db.csv'),index_col=0)

# Getting AA sequence from Uniprot ID
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

dtc_targets = pd.DataFrame( dtc_db.target_id.unique(),
    get_aaseq(dtc_db.target_id.unique()), columns = ['UniProtID','Sequence'])
dtc_compounds = dtc_db.compound_id.unique()
