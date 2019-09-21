import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from Add_category import multi_column
from itertools import cycle

def roc_calculation(fpr, tpr, roc_auc, y_test, y_score, n_species):
    for species in range(n_species):
        fpr[species], tpr[species], _ = roc_curve(y_test[:, species],y_score[:, species])
        roc_auc[species] = auc(fpr[species], tpr[species])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


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
for k in data:
    data[k] = pd.read_parquet(data[k])

#message of ongoing compilation
print('adding category to dataframes')

# add category to dataframes
multi_column(data)

# training columns
train_test_columns = ['p', 'pTPC', 'ITSclsMap',
                    'dEdxITS', 'NclusterPIDTPC', 'dEdxTPC']

# training dataframe
train_df = pd.concat([data[key].iloc[:20000] for key in data], ignore_index=True)

# testing dataframe
test_df =  pd.concat([data[key].iloc[20000:] for key in data], ignore_index=True)

# train and test dataframe for classifier
x_train_df = train_df[train_test_columns]
y_train_df = train_df[keys]
x_test_df  = test_df[train_test_columns]
y_test_df  = test_df[keys]

#classifier
clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=3,n_estimators=200, learning_rate=0.2))

#training classifier
clf.fit(x_train_df,y_train_df)

#prediction of classifier
y_pred_train = clf.predict_proba(x_train_df)
y_pred_test  = clf.predict_proba(x_test_df)

#confusion matrixes
conf_matr_train = confusion_matrix(y_train_df,y_pred_train)
conf_matr_test  = confusion_matrix(y_test_df,y_pred_test)

#normalize confusion matrix
conf_matr_train = conf_matr_train.astype('float') / conf_matr_train.sum(axis=1)[:, np.newaxis]
conf_matr_test = conf_matr_test.astype('float') / conf_matr_test.sum(axis=1)[:, np.newaxis]

#correlation matrix
corr_df = x_train_df.corr(method='pearson')

#fpr,tpr,roc_auc_micro
fpr_train = dict()
tpr_train = dict()
roc_auc_train = dict()
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()

roc_calculation(fpr_train,tpr_train,roc_auc_train,y_train_df,y_pred_train,len(keys))
roc_calculation(fpr_test,tpr_test,roc_auc_test,y_test_df,y_pred_test,len(keys))

#figure for plot
f1 = plt.figure()

#plot confusion matrix train
ax1 = plt.subplot(nrows=1,ncols=3,index=1)
im1 = ax1.imshow(conf_matr_train, interpolation='nearest', cmap=plt.get_cmap('Blues'))
ax1.figure.colorbar(im1, ax=ax1)
# We want to show all ticks...
ax1.set(xticks=np.arange(conf_matr_train.shape[1]),
       yticks=np.arange(conf_matr_train.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=keys, yticklabels=keys,
       title='Confusion matrix of train',
       ylabel='True label',
       xlabel='Predicted label')

#plot confusion matrix test
ax2 = plt.subplot(nrows=1,ncols=3,index=2)
im2 = ax2.imshow(conf_matr_test, interpolation='nearest', cmap=plt.get_cmap('Greens'))
ax2.figure.colorbar(im2, ax=ax2)
# We want to show all ticks...
ax2.set(xticks=np.arange(conf_matr_test.shape[1]),
       yticks=np.arange(conf_matr_test.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=keys, yticklabels=keys,
       title='Confusion matrix of test',
       ylabel='True label',
       xlabel='Predicted label')

#plot confusion matrix test
ax3 = plt.subplot(nrows=1,ncols=3,index=3)
im3 = ax3.imshow(corr_df, plt.get_cmap('Reds'))
ax3.figure.colorbar(im3, ax=ax3)
# We want to show all ticks...
ax3.set(xticks=np.arange(corr_df.shape[1]),
       yticks=np.arange(corr_df.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=keys, yticklabels=keys,
       title='Correlation matrix',
       )

f1.tight_layout()

plt.savefig('correlation_and_confusion_matrixes.pdf')

#new figure for roc auc
f2 = plt.figure()
#train roc auc
plt.subplot(nrows=1,ncols=2,index=1)
plt.title = 'Train ROC-AUC'
plt.plot(fpr_train["micro"], tpr_train["micro"], color='gold', linestyle = ':',
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_train["micro"]))
colors = cycle(['lime', 'mediumturquoise', 'maroon','orangered','darkblue'])
for ind,color in zip(range(len(keys)), colors):
    plt.plot(fpr_train[ind], tpr_train[ind], color=color,
             label='ROC curve of' + keys[ind] +' (area = {0:0.2f})'
             ''.format(roc_auc_train[ind]))
#test roc auc
plt.subplot(nrows=1,ncols=2,index=2)
plt.title = 'Test ROC-AUC'
plt.plot(fpr_test["micro"], tpr_test["micro"], color='gold', linestyle = ':',
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_test["micro"]))
colors = cycle(['lime', 'mediumturquoise', 'maroon','orangered','darkblue'])
for ind,color in zip(range(len(keys)), colors):
    plt.plot(fpr_test[ind], tpr_test[ind], color=color,
             label='ROC curve of' + keys[ind] +' (area = {0:0.2f})'
             ''.format(roc_auc_test[ind]))

f2.tight_layout()

plt.savefig('ROC_AUC.pdf')

print('done')