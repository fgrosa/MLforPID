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

# function for calculation of roc auc micro average
def roc_calculation(fpr, tpr, roc_auc, y_test, y_score, n_species):
    for species in range(n_species):
        fpr[species], tpr[species], _ = roc_curve(
            y_test[:, species], y_score[:, species])
        roc_auc[species] = auc(fpr[species], tpr[species])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

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
    data[key] = pd.read_parquet(data[key]).iloc[0:80000]
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
train_df = pd.concat([data[key].iloc[0:40000]
                      for key in data], ignore_index=True)

# testing dataframe
test_df = pd.concat([data[key].iloc[40000:80000]
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

# normalize confusion matrix
conf_matr_train = conf_matr_train.astype(
    'float32') / conf_matr_train.sum(axis=1)[:, np.newaxis]
conf_matr_test = conf_matr_test.astype(
    'float32') / conf_matr_test.sum(axis=1)[:, np.newaxis]

# fpr,tpr,roc_auc_micro
fpr_train = dict()
tpr_train = dict()
roc_auc_train = dict()
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()
y_test_multi = label_binarize(y_test_df, classes=range(len(keys)))
y_train_multi = label_binarize(y_train_df, classes=range(len(keys)))


roc_calculation(fpr_train, tpr_train, roc_auc_train,
                y_train_multi, y_proba_train, len(keys))
roc_calculation(fpr_test, tpr_test, roc_auc_test,
                y_test_multi, y_proba_test, len(keys))

print('start creating figures for confusion matrix, ROC AUC, feature importance and distributions of probability')

# figure for plot
f1 = plt.figure(figsize=[20, 5])

# plot confusion matrix train
ax1 = plt.subplot(1, 2, 1)
im1 = ax1.imshow(conf_matr_train,
                 cmap=plt.get_cmap('Reds'), interpolation='nearest')
ax1.figure.colorbar(im1, ax=ax1)
# show all ticks
ax1.set(xticks=np.arange(conf_matr_train.shape[1]),
        yticks=np.arange(conf_matr_train.shape[0]),
        # label them with the respective list entries
        xticklabels=keys, yticklabels=keys,
        ylabel='True label',
        xlabel='Predicted label')
ax1.set_title('Confusion matrix of train', y=1.08)
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)

# Rotate the tick labels and set their alignment.
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")
plt.setp(ax1.get_yticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")

#add values in the middle of the cell
for i in range(conf_matr_test.shape[0]):
        for j in range(conf_matr_train.shape[1]):
            ax1.text(j, i, format(conf_matr_train[i, j], '.4f'),
                    ha="center", va="center",
                    color="black", fontsize=12)

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

print('confusion matrix train done')

# plot confusion matrix test
ax2 = plt.subplot(1, 2, 2)
im2 = ax2.imshow(conf_matr_test,
                 cmap=plt.get_cmap('Greens'), interpolation='nearest')
ax2.figure.colorbar(im2, ax=ax2)
ax2.set(xticks=np.arange(conf_matr_test.shape[1]),
        yticks=np.arange(conf_matr_test.shape[0]),
        xticklabels=keys, yticklabels=keys,
        ylabel='True label',
        xlabel='Predicted label')
ax2.set_title('Confusion matrix of test', y=1.08)
bottom_2, top_2 = ax2.get_ylim()
ax2.set_ylim(bottom_2 + 0.5, top_2 - 0.5)

# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")
plt.setp(ax2.get_yticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")

#add values in the middle of the cell
for i in range(conf_matr_test.shape[0]):
        for j in range(conf_matr_test.shape[1]):
            ax2.text(j, i, format(conf_matr_test[i, j], '.4f'),
                    ha="center", va="center",
                    color="black", fontsize=12)

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

f1.tight_layout()

print('confusion matrix test done')

# plt.savefig('confusion_matrix_Multiclass.pdf')

# new figure for roc auc
f2 = plt.figure(figsize=[10, 5], constrained_layout=True)

# train roc auc
plt.subplot(1, 2, 1)
plt.title = 'Train ROC-AUC'
colors = cycle(['lightcoral', 'khaki', 'yellowgreen', 'lightblue', 'lightsteelblue'])
for ind, color in zip(range(len(keys)), colors):
    plt.plot(fpr_train[ind], tpr_train[ind], color=color,
             label='ROC curve of ' + keys[ind] + ' (area = {0:0.4f})'
             ''.format(roc_auc_train[ind]))
plt.plot(fpr_train["micro"], tpr_train["micro"], color='black', linestyle=':',
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc_train["micro"]))
plt.xlabel('Background Efficiency')
plt.ylabel('Signal Efficiency')
plt.legend()

print('ROC AUC train done')

# test roc auc
plt.subplot(1, 2, 2)
plt.title = 'Test ROC-AUC'
colors = cycle(['lightcoral', 'khaki', 'yellowgreen', 'lightblue', 'lightsteelblue'])
for ind, color in zip(range(len(keys)), colors):
    plt.plot(fpr_test[ind], tpr_test[ind], color=color,
             label='ROC curve of ' + keys[ind] + ' (area = {0:0.4f})'
             ''.format(roc_auc_test[ind]))
plt.plot(fpr_test["micro"], tpr_test["micro"], color='black', linestyle=':',
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc_test["micro"]))
plt.xlabel('Background Efficiency')
plt.ylabel('Signal Efficiency')
plt.legend()

f2.tight_layout()

# plt.savefig('ROC_AUC_Multiclass.pdf')

# new figure for feature importance
f3 = plt.figure(figsize=[10, 5])

# plot feature importance
ax3 = plt.subplot(1,1,1)
feat_importances = pd.Series(clf.feature_importances_, index = x_train_df.columns)
feat_importances.plot(kind='barh', title = 'Feature Importance')

# plt.savefig('Feature Importance.pdf')

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
    plt.ylabel('log(entries)')
    plt.xlim(0,1)
    plt.legend(loc='best')
    # fighist.savefig('probability_distribution_of_{0}.pdf'.format(name))

plt.show()