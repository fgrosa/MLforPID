import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from Add_category import single_column
from itertools import cycle
from sklearn.preprocessing import label_binarize
from xgboost import plot_importance

# flags
PARSER = argparse.ArgumentParser(description='Arguments to pass')
PARSER.add_argument('dir', metavar='text', default='.',
                    help='flag to search in data directory, remember put all paths!')
ARGS = PARSER.parse_args()

# list_files necessary
list_files = [f for f in os.listdir(ARGS.dir) if (
    'data.parquet.gzip' in f and not f.startswith("._"))]
list_files.remove('kaons_fromTOF_data.parquet.gzip')
list_files.remove('He3_fromTOFTPC_data.parquet.gzip')
list_files.remove('triton_fromTOFTPC_data.parquet.gzip')
os.chdir(ARGS.dir)

# keys for dictionary
keys = list(map(lambda x: x.split('_')[0], list_files))

# dictionary for data
data = dict(zip(keys, list_files))

# create dataframe for dictionaries
for key in data:
    data[key] = pd.read_parquet(data[key]).iloc[0:800]
    header = data[key].select_dtypes(include=[np.float64]).columns
    # change dtype of column with float64
    data[key][header] = data[key][header].astype('float32')

# message of ongoing compilation
print('adding category to dataframes')

# add category to dataframes
single_column(data)

# training columns
train_test_columns = ['p', 'pTPC', 'ITSclsMap',
    'dEdxITS', 'NclusterPIDTPC', 'dEdxTPC']

# training dataframe
train_df = pd.concat([data[key].iloc[0:400]
                      for key in data], ignore_index=True)

# testing dataframe
test_df = pd.concat([data[key].iloc[400:800]
                     for key in data], ignore_index=True)

# train and test dataframe for classifier
x_train_df = train_df[train_test_columns]
y_train_df = train_df['category']
x_test_df = test_df[train_test_columns]
y_test_df = test_df['category']

# classifier
clf = XGBClassifier(
    n_jobs=-1, max_depth=3, n_estimators=200, learning_rate=0.2)

# training classifier
clf.fit(x_train_df, y_train_df)

# prediction of classifier
y_pred_train = clf.predict(x_train_df)
y_pred_test  = clf.predict(x_test_df)   

# probabilities of classifier
y_proba_train = clf.predict_proba(x_train_df)
y_proba_test  = clf.predict_proba(x_test_df)

# confusion matrixes
conf_matr_train = confusion_matrix(y_train_df.values,
    y_pred_train)
conf_matr_test = confusion_matrix(y_test_df.values,
    y_pred_test)

# fpr,tpr,roc_auc_micro
fpr_train = dict()
tpr_train = dict()
roc_auc_train = dict()
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()
y_test_multi = label_binarize(y_test_df, classes=range(len(keys)))
y_train_multi = label_binarize(y_train_df, classes=range(len(keys)))

#plot distribution of probabilities

#color dictionary
col = {'electrons': 'blue', 'pi': 'orangered', 'kaons': 'red',
    'protons': 'green', 'deuterons': 'grey'}

#adding distribution prob.
for prob,key in enumerate(keys):
    train_df['prob_{0}'.format(key)] = y_proba_train[:, prob]
    test_df['prob_{0}'.format(key)] = y_proba_test[:, prob]

for prob_key,name in enumerate(keys):
    fighist = plt.figure(figsize=[10,8])
    for key in keys:
       #plot histogram
        hist, bins, _ = plt.hist(train_df.loc[train_df['category'] == prob_key]['prob_{0}'.format(key)], color = col[key],
        alpha = 1, bins = 100, histtype='step', ec = col[key], density=True, label = '{0}_train'.format(key), log=True)
        center = (bins[:-1] + bins[1:]) / 2
        plt.fill_between(center, [1.e-4 for b in range(len(center))], hist,color = col[key], alpha=0.25, interpolate=True)
        plt.ylim(1.e-4,2.e2)
        #error_bar
        hist, bins = np.histogram(test_df.loc[test_df['category'] == prob_key]['prob_{0}'.format(key)].values, bins = 100, density = True)
        scale = len(test_df) / sum(hist)
        err = np.sqrt(hist * scale) / scale
        plt.errorbar(center, hist, yerr=err, fmt='o', c=col[key], label = '{0}_test'.format(key))
    plt.xlabel('probability to be {0}'.format(name))
    plt.ylabel('entries')
    plt.xlim(0,1)
    plt.legend(loc='best')
    fighist.savefig('probability_distribution_of_{0}.pdf'.format(name))

plt.show()
print('done')