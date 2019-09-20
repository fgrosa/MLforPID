import numpy as np
import pandas as pd
from math import sqrt
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from Add_category import multi_column
from itertools import product

# flags
PARSER = argparse.ArgumentParser(description='Arguments to pass')
PARSER.add_argument('dir', metavar='text', default='.',
                    help='flag to search in data directory, remember put all paths!')
ARGS = PARSER.parse_args()

# list_files
list_files = [f for f in os.listdir(ARGS.dir) if 'data.parquet.gzip' in f]
list_files.remove('kaons_fromTOF_data.parquet.gzip')
list_files.remove('He3_fromTOFTPC_data.parquet.gzip')
list_files.remove('triton_fromTOFTPC_data.parquet.gzip')
os.chdir(ARGS.dir)

# keys for dictionary
keys = list(map(lambda x: x.split('_')[0], list_files))

# dictionary for data
data = dict(zip(keys, list_files))

# create dataframe for dictionaries
for keys in data:
    data[keys] = pd.read_parquet(data[keys])

#message of ongoing compilation
print('adding category to dataframes')

# add category to dataframes
multi_column(data)

# training columns
training_columns = ['p', 'pTPC', 'ITSclsMap',
                    'dEdxITS', 'NclusterPIDTPC', 'dEdxTPC']

# training dataframe
training_df = pd.concat([data[keys].iloc[:5000]
                         for keys in data], ignore_index=True)

# traing data with training columns
X_df = training_df[training_columns]
Y_df = training_df.filter(like='category', axis=1)

