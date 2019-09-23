import numpy as np
import pandas as pd
from math import sqrt
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
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

# parameters for grid search
max_depth = [2, 3]  # ,3,5,8]
n_estimators = [100]  # ,200,500]
learning_rate = [0.1]  # ,0.2,0.3]

# all combination of parameter
parameters = product(max_depth, n_estimators, learning_rate)

# number of folds
n_folds = 5

# folding for cross validation
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

#creating dataframe for parameters values
paramsvalue = []

print('grid search started')

#manual grid search
for n,param in enumerate(parameters):
    scores = np.array()
    for train_in,val_in in kf.split(len(training_df)):
        # onevsrest classifier
        clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=param[0], n_estimators=param[1], learning_rate=[2]))
        clf.fit(X_df[train_in],Y_df[train_in])
        score = roc_auc_score(Y_df[val_in],clf.predict(X_df[val_in]),average = 'micro')
        scores.append(score)	
    mean = np.mean(scores)
    std  = np.std(scores)
    paramsvalue.append((index = {5},max_depth = {0}, n_estimator = {1}, learning_rate = {2}, roc_auc_micro_score = {3} +/- {4}).format(
    param[0],param[1],param[2],mean,std,n))

# list of columns for df
columns_df = ['index', 'max_depth', 'n_estimator',
               'learning_rate', 'roc_auc_micro']
# dataframe of the results of grid search
df_results = pd.DataFrame(paramsvalue, columns=columns_df)
# number of combination of parameters
n_set_params = int(len(df_results)/n_folds)
#new columns for dataframe
df_results['root_mean_square'] = 0
df_results['mean_roc_auc'] = 0

# conversion to parquet
df_results.to_parquet('results_grid_search.parquet.gzip', compression='gzip')
print('dataframe of the results of grid search saved')

# plotting datas of average_roc_auc_micro in function of index
plt.plot(df_results['index'], df_results['mean_roc_auc'])
plt.fill_between(df_results['index'], df_results['mean_roc_auc']-df_results['root_mean_square'],
                 df_results['mean_roc_auc']+df_results['root_mean_square'], alpha=0.3)
plt.title('mean_roc_auc - index')
plt.xlabel('index')
plt.ylabel('mean_roc_auc')
#save file
plt.savefig('gridsearch_multiclass_results.pdf')
#last line of finish
print('all done')
