import numpy as np
import pandas as pd
import os
import argparse
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from Add_category import add_category
from itertools import product

# flags
PARSER = argparse.ArgumentParser(description='Arguments to pass')
PARSER.add_argument('dir', metavar='text', default='.',
    help='flag to search in data directory, remember put all paths!')
ARGS = PARSER.parse_args()

# files
files = [f for f in os.listdir(ARGS.dir) if 'data.parquet.gzip' in f]
files.remove('kaons_fromTOF_data.parquet.gzip')
files.remove('He3_fromTOFTPC_data.parquet.gzip')
files.remove('triton_fromTOFTPC_data.parquet.gzip')
os.chdir(ARGS.dir)

# keys for dictionary
keys = list(map(lambda x: x.split('_')[0], files))
# dictionary for data
data = dict(zip(keys, files))

# create dataframe for dictionaries
for keys in data:
    data[keys] = pd.read_parquet(data[keys])

print('adding category to dataframes')
#add category to dataframes
add_category(data)

#training columns
training_columns = ['p','pTPC','ITSclsMap','dEdxITS','NclusterPIDTPC','dEdxTPC']

#training dataframe
training_df = pd.concat([data[keys].iloc[:5000] for keys in data],ignore_index = True)

#traing data with training columns
X_df = training_df[training_columns]
Y_df = training_df['category']

#parameters for grid search
max_depth=[2,3,5,8]
n_estimators=[100,200,500]
#all combination of parameter
parameters = product(max_depth,n_estimators)

#folding for cross validation
kf = KFold(n_splits=5)

#create files for storing scores
f = open('roac_auc_score.txt','w+')
f.write('ROC_AUC_SCORES \n')

#manual grid search
for param in parameters:
    scores = np.array()
    for train_in,val_in in kf.split(len(training_df)):
        # onevsrest classifier
        clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=param[0], n_estimators=param[1]))
        clf.fit(X_df[train_in],Y_df[train_in])
        score = roc_auc_score(Y_df[val_in],clf.predict(X_df[val_in]),average = 'micro')
        scores.append(score)
    mean = np.mean(scores)
    std  = np.std(scores)
    f.write('max_depth = {0}, n_estimator = {1}, roc_auc_micro_score = {2} +/- {3}'.format(
        param[0],param[1],mean,std))
    del scores


