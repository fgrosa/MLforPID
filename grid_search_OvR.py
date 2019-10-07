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
for key in data:
    data[key] = pd.read_parquet(data[key])

print('adding category to dataframes')

# add category to dataframes
multi_column(data)

# training columns
training_columns = ['p', 'pTPC', 'ITSclsMap',
                    'dEdxITS', 'NclusterPIDTPC', 'dEdxTPC']

# training dataframe
training_df = pd.concat([data[key].iloc[:5000]
                         for key in data], ignore_index=True)

# traing data with training columns
X_df = training_df[training_columns]
Y_df = training_df[keys]

# parameters for grid search
max_depth = [2,3,5,8]
n_estimators = [100,200,500]
learning_rate = [0.1,0.2,0.3]  

# all combination of parameter
parameters = product(max_depth, n_estimators, learning_rate)

# number of folds
n_folds = 5

# folding for cross validation
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# list for dataframe of results and parameters
list_for_df = []

print('grid search started')

# manual grid search
for ipar, param in enumerate(parameters):
    for index in kf.split(training_df):
        # onevsrest classifier
        clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=param[0],
                                                n_estimators=param[1], learning_rate=param[2]))
        x_train = X_df.iloc[index[0]]
        y_train = Y_df.iloc[index[0]]
        x_test = X_df.iloc[index[1]]
        y_test = Y_df.iloc[index[1]]
        y_score = clf.fit(x_train, y_train).predict_proba(x_test)
        n_classes = range(len(data))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for cat in n_classes:
            fpr[cat], tpr[cat], _ = roc_curve(
                y_test.loc[:, keys[cat]].values, y_score[:, cat])
            roc_auc[cat] = auc(fpr[cat], tpr[cat])
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_test.loc[:, keys[0]: keys[len(n_classes)-1]].values.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # temporary list
        tmp_list = [ipar, param[0], param[1], param[2], roc_auc['micro']]
        list_for_df.append(tmp_list)

# list of columns for df
columns_df = ['index', 'max_depth', 'n_estimator',
               'learning_rate', 'roc_auc_micro']
# dataframe of the results of grid search
df_results = pd.DataFrame(list_for_df, columns=columns_df)
# number of combination of parameters
n_set_params = int(len(df_results)/n_folds)
#new columns for dataframe
df_results['root_mean_square'] = 0
df_results['mean_roc_auc'] = 0

# calculation of average and
for ind_df in range(n_set_params):
    #indexes
    start_ind = ind_df*n_folds
    end_ind = (ind_df+1)*n_folds
    #average of roc_auc_micro
    average_roc = df_results.loc[start_ind:end_ind, 'roc_auc_micro'].mean()
    df_results.loc[start_ind:end_ind, 'mean_roc_auc'] = average_roc

    # calculate rms mean
    df_results['residuals'] = (
        df_results.loc[start_ind:end_ind, 'roc_auc_micro'] - average_roc)**2
    rms = sqrt(df_results.loc[start_ind:end_ind, 'residuals'].mean())
    df_results.loc[start_ind:end_ind, 'root_mean_square'] = rms

#eliminate the column residuals which is now unneccesary
df_results = df_results.drop(columns='residuals')

# conversion to parquet
df_results.to_parquet('results_grid_search_OvsR.parquet.gzip', compression='gzip')
print('dataframe of the results of grid search saved')

# plotting datas of average_roc_auc_micro in function of index
plt.plot(df_results['index'], df_results['mean_roc_auc'])
plt.fill_between(df_results['index'], df_results['mean_roc_auc']-df_results['root_mean_square'],
                 df_results['mean_roc_auc']+df_results['root_mean_square'], alpha=0.3)
plt.title('mean_roc_auc - index')
plt.xlabel('index')
plt.ylabel('mean_roc_auc')
#save file
plt.savefig('gridsearch_OvsR_results.pdf')
#last line of finish
print('all done')
