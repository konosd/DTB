import os

# #########
# Initializer woohoooo
# #########

data_dir = os.path.join(os.getcwd(),'data')
source_dir = os.path.join(os.getcwd(),'code')

pretty_data = os.path.join(source_dir,'pretty_data.py')
print('Parsing and tidying the DTC database, sir.')
exec(open(pretty_data).read())