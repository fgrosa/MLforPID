import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from Add_category import multi_column
from itertools import cycle
from sklearn.preprocessing import LabelBinarizer

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
    data[key] = pd.read_parquet(data[key]).iloc[:10000]
    header = data[key].select_dtypes(include=[np.float64]).columns
    # change dtype of column with float64
    data[key][header] = data[key][header].astype('float32')

# message of ongoing compilation
print('adding category to dataframes')

# add category to dataframes
multi_column(data)

# training columns
train_test_columns = ['p', 'pTPC', 'ITSclsMap',
    'dEdxITS', 'NclusterPIDTPC', 'dEdxTPC']

# training dataframe
train_df = pd.concat([data[key].iloc[:5000]
                      for key in data], ignore_index=True)

# testing dataframe
test_df = pd.concat([data[key].iloc[5000:10000]
                     for key in data], ignore_index=True)

# train and test dataframe for classifier
x_train_df = train_df[train_test_columns]
y_train_df = train_df[keys]
x_test_df = test_df[train_test_columns]
y_test_df = test_df[keys]

# classifier
clf = OneVsRestClassifier(XGBClassifier(
    n_jobs=-1, max_depth=3, n_estimators=200, learning_rate=0.2))

# training classifier
clf.fit(x_train_df, y_train_df)

# prediction of classifier
y_pred_train = clf.predict(x_train_df)
y_pred_test  = clf.predict(x_test_df)

# probabilities of classifier
y_proba_train = clf.predict_proba(x_train_df)
y_proba_test  = clf.predict_proba(x_test_df)

# convert multi-label to multi-class
lb = LabelBinarizer()
lb.fit(range(len(keys)))

# confusion matrixes
conf_matr_train = confusion_matrix(lb.inverse_transform(y_train_df.values),
    lb.inverse_transform(y_pred_train))
conf_matr_test = confusion_matrix(lb.inverse_transform( 
    y_test_df.values), lb.inverse_transform(y_pred_test))

# normalize confusion matrix
conf_matr_train = conf_matr_train.astype(
    'float32') / conf_matr_train.sum(axis=1)[:, np.newaxis]
conf_matr_test = conf_matr_test.astype(
    'float32') / conf_matr_test.sum(axis=1)[:, np.newaxis]

# correlation matrix
corr_df = x_train_df.corr(method='pearson')

# fpr,tpr,roc_auc_micro
fpr_train = dict()
tpr_train = dict()
roc_auc_train = dict()
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()

roc_calculation(fpr_train, tpr_train, roc_auc_train,
                y_train_df.values, y_proba_train, len(keys))
roc_calculation(fpr_test, tpr_test, roc_auc_test,
                y_test_df.values, y_proba_test, len(keys))

# figure for plot
f1 = plt.figure(figsize=[15, 15], constrained_layout=True)

# plot confusion matrix train
ax1 = plt.subplot(1, 2, 1)
im1 = ax1.imshow(conf_matr_train, interpolation='nearest',
                 cmap=plt.get_cmap('Reds'))
ax1.figure.colorbar(im1, ax=ax1)
# show all ticks
ax1.set(xticks=np.arange(conf_matr_train.shape[1]),
        yticks=np.arange(conf_matr_train.shape[0]),
        # label them with the respective list entries
        xticklabels=keys, yticklabels=keys,
        title='Confusion matrix of train',
        ylabel='True label',
        xlabel='Predicted label')
# Rotate the tick labels and set their alignment.
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")

#add values in the middle of the cell
for i in range(conf_matr_test.shape[0]):
        for j in range(conf_matr_train.shape[1]):
            ax1.text(j, i, format(conf_matr_train[i, j], '.2f'),
                    ha="center", va="center",
                    color="black", fontsize=4)

plt.xticks(fontsize = 5)
plt.yticks(fontsize = 5)

# plot confusion matrix test
ax2 = plt.subplot(1, 2, 2)
im2 = ax2.imshow(conf_matr_test, interpolation='nearest',
                 cmap=plt.get_cmap('Greens'))
ax2.figure.colorbar(im2, ax=ax2)
ax2.set(xticks=np.arange(conf_matr_test.shape[1]),
        yticks=np.arange(conf_matr_test.shape[0]),
        xticklabels=keys, yticklabels=keys,
        title='Confusion matrix of test',
        ylabel='True label',
        xlabel='Predicted label')
# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")

#add values in the middle of the cell
for i in range(conf_matr_test.shape[0]):
        for j in range(conf_matr_test.shape[1]):
            ax2.text(j, i, format(conf_matr_test[i, j], '.2f'),
                    ha="center", va="center",
                    color="black", fontsize=4)

plt.xticks(fontsize = 5)
plt.yticks(fontsize = 5)

f1.tight_layout()

plt.savefig('confusion_matrix_OvR.pdf')

# new figure for roc auc
f2 = plt.figure(figsize=[15, 15], constrained_layout=True)
# train roc auc
plt.subplot(1, 2, 1)
plt.title = 'Train ROC-AUC'
plt.plot(fpr_train["micro"], tpr_train["micro"], color='gold', linestyle=':',
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_train["micro"]))
colors = cycle(['lime', 'mediumturquoise', 'maroon', 'orangered', 'darkblue'])
for ind, color in zip(range(len(keys)), colors):
    plt.plot(fpr_train[ind], tpr_train[ind], color=color,
             label='ROC curve of' + keys[ind] + ' (area = {0:0.2f})'
             ''.format(roc_auc_train[ind]))

plt.legend()

# test roc auc
plt.subplot(1, 2, 2)
plt.title = 'Test ROC-AUC'
plt.plot(fpr_test["micro"], tpr_test["micro"], color='black', linestyle=':',
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_test["micro"]))
colors = cycle(['lightcoral', 'khaki', 'yellowgreen', 'lightblue', 'lightsteelblue'])
for ind, color in zip(range(len(keys)), colors):
    plt.plot(fpr_test[ind], tpr_test[ind], color=color,
             label='ROC curve of' + keys[ind] + ' (area = {0:0.2f})'
             ''.format(roc_auc_test[ind]))

plt.legend()

f2.tight_layout()

plt.savefig('ROC_AUC_OvR.pdf')

# new figure
f3 = plt.figure(figsize=[15, 15], constrained_layout=True)
# plot confusion matrix test
ax3 = plt.subplot(1, 1, 1)
im3 = ax3.imshow(corr_df, plt.get_cmap('coolwarm'))
ax3.figure.colorbar(im3, ax=ax3,)
ax3.set(xticks=np.arange(corr_df.shape[1]),
        yticks=np.arange(corr_df.shape[0]),
        xticklabels=train_test_columns, yticklabels=train_test_columns,
        title='Correlation matrix',
        )
# Rotate the tick labels and set their alignment.
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")

#add values in the middle of the cell
for i in range(corr_df.values.shape[0]):
        for j in range(corr_df.values.shape[1]):
            ax3.text(j, i, format(corr_df.values[i, j], '.2f'),
                    ha="center", va="center",
                    color="black",fontsize=4)

plt.xticks(fontsize = 5)
plt.yticks(fontsize = 5)

plt.savefig('correlation_matrix_OvR.pdf')

# plot new confusion matrix for multilabel
new_conf_matr_train = multilabel_confusion_matrix(y_train_df, y_pred_train)
new_conf_matr_test = multilabel_confusion_matrix(y_test_df, y_pred_test)
# normalize confusion matrix
new_conf_matr_train = new_conf_matr_train.astype(
    'float32') / new_conf_matr_train.sum(axis=1)[:, np.newaxis]
new_conf_matr_test = new_conf_matr_test.astype(
    'float32') / new_conf_matr_test.sum(axis=1)[:, np.newaxis]
conf_matr_list = [new_conf_matr_train, new_conf_matr_test]

# new figure and plot confusion matrixes
f4, axes = plt.subplots(nrows=2, ncols=5, figsize=[35, 35], constrained_layout=True)
for i in range(2):
    for j in range(5):
        ax = axes[i, j]
        ticks = [keys[j], 'background']
        cm = conf_matr_list[i][j]
        im = ax.imshow(cm, plt.get_cmap('Blues'))
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks = np.arange(cm.shape[1]), yticks = np.arange(cm.shape[0]), xticklabels = ticks, yticklabels = ticks)
        for ind_i in range(cm.shape[0]):
            for ind_j in range(cm.shape[1]):
                ax.text(ind_j, ind_i, format(cm[ind_i, ind_j], '.2f'),
                    ha="center", va="center",
                    color="black", fontsize=4)

plt.xticks(fontsize = 5)
plt.yticks(fontsize = 5)

f4.tight_layout()

plt.savefig('confusion_matrix_of_species_OvrR.pdf')

#adding distribution prob.
for key,prob in zip(keys,range(len(keys))):
    train_df['prob_{0}'.format(key)] = y_proba_train[:, prob]
    test_df['prob_{0}'.format(key)] = y_proba_test[:, prob]

#plot distribution of probabilities
for key in keys:
    fighist = plt.figure()
    plt.hist(train_df[key]['prob_{0}'.format(key)], color = 'r',
    alpha =0.5, bins = 50, histtype='stepfilled', density=True,
    label = '{0}_train'.format(key))
    plt.hist(test_df[key]['prob_{0}'.format(key)], color = 'b',
    alpha =0.5, bins = 50, histtype='stepfilled', density=True, 
    label = '{0}_test'.format(key))
    hist, bins = test_df.hist(column = 'prob_{0}'.format(key), bins = 50, density = True )
    scale = len(test_df) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b')
    fighist.savefig('probability_distribution_of_{0}_OvR.pdf'.format(key))

plt.show()

print('done')
