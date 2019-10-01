import os
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from Add_category import multi_column
from itertools import cycle
#function for later
def roc_calculation(fpr, tpr, roc_auc, y_test, y_score, n_species):
    for species in range(n_species):
        fpr[species], tpr[species], _ = roc_curve(
            y_test[:, species], y_score[:, species])
        roc_auc[species] = auc(fpr[species], tpr[species])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# flags
PARSER = argparse.ArgumentParser(description='Arguments to pass')
PARSER.add_argument('data_dir', metavar='text', default='.',
    help='flag to search in data directory, remember put all paths!')
PARSER.add_argument('mc_dir', metavar='text', default='.',
    help='flag to search in MC directory, remember put all paths!')
ARGS = PARSER.parse_args()

# data_files necessary
data_files = [f for f in os.listdir(ARGS.data_dir) if (
    'data.parquet.gzip' in f and not f.startswith("._"))]
data_files.remove('kaons_fromTOF_data.parquet.gzip')
data_files.remove('He3_fromTOFTPC_data.parquet.gzip')
data_files.remove('triton_fromTOFTPC_data.parquet.gzip')
os.chdir(ARGS.data_dir)

# keys for data
data_keys = list(map(lambda x: x.split('_')[0], data_files))

# dictionary for data
df_data = dict(zip(data_keys, data_files))

# create dataframe for data
for key in df_data:
    df_data[key] = pd.read_parquet(df_data[key])
    header = df_data[key].select_dtypes(include=[np.float64]).columns
    # change dtype of column with float64 to float16
    df_data[key][header] = df_data[key][header].astype('float16')

#add category to df
multi_column(df_data)

#load model saved
clf = pickle.load(open('OvsR_trained.pkl', 'rb'))

# MC_files necessary
mc_files = [f for f in os.listdir(ARGS.mc_dir) if (
    'MC.parquet.gzip' in f and not f.startswith("._"))]
mc_files.remove('kaons_fromTOF_MC.parquet.gzip')
mc_files.remove('He3_fromTOFTPC_MC.parquet.gzip')
mc_files.remove('triton_fromTOFTPC_MC.parquet.gzip')
os.chdir(ARGS.mc_dir)

# pdgcode
pdg = {'e': 11, 'pi': 211, 'kaons': 321, 'p': 2212,
    'He3': 1000020030, 'triton': 1000010030, 'deuterons': 1000010020}

# keys for mc
mc_keys = list(map(lambda x: x.split('_')[0], mc_files))

#dictionary for mc
df_mc = dict(zip(mc_keys, mc_files))

# create dataframe for mc
for key in df_mc:
    df_mc[key] = pd.read_parquet(df_mc[key])
    header = df_mc[key].select_dtypes(include=[np.float64]).columns
    # change dtype of column with float64 to float16
    df_mc[key][header] = df_mc[key][header].astype('float16')
    df_mc[key] = df_mc[key].query('PDGcode == {0}'.format(pdg[key]))

#columns used to fit
target_columns = ['p', 'pTPC', 'ITSclsMap',
    'dEdxITS', 'NclusterPIDTPC', 'dEdxTPC']

#all dataframe for testing and roc auc
df_data_test = pd.concat([df_data[key].iloc[int(len(df_data[key])/5):]
                     for key in df_data], ignore_index=True)

#predict data based on the target columns
df_data_pred_proba = clf.predict_proba(df_data_test[target_columns])

#dataframe mc for test
df_mc_test = pd.concat([df_mc[key] for key in df_mc], ignore_index=True)

#predict mc based on target column
df_mc_pred_proba = clf.predict_proba(df_mc[target_columns])

#fpr ,tpr and roc_auc 
fpr_data = dict()
tpr_data = dict()
fpr_mc = dict()
tpr_mc = dict()
roc_auc_data = dict()
roc_auc_mc = dict()

#calculation of roc_auc_data
roc_calculation(fpr_data, tpr_data, roc_auc_data,
    df_data_test.values, df_data_pred_proba, len(data_keys))
roc_calculation(fpr_mc, tpr_mc, roc_auc_mc,
    df_mc_test.values, df_mc_pred_proba, len(mc_keys))

#roc auc curve
# new figure for roc auc
f = plt.figure(figsize=[10, 5], constrained_layout=True)
# train roc auc
plt.subplot(1, 2, 1)
plt.title = 'Data ROC-AUC'
colors = cycle(['lightcoral', 'khaki', 'yellowgreen', 'lightblue', 'lightsteelblue'])
for ind, color in enumerate(colors):
    plt.plot(fpr_data[ind], tpr_data[ind], color=color,
             label='ROC curve of ' + data_keys[ind] + ' (area = {0:0.4f})'
             ''.format(roc_auc_data[ind]))
plt.plot(fpr_data["micro"], tpr_data["micro"], color='black', linestyle=':',
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc_data["micro"]))
plt.xlabel('background efficiency')
plt.ylabel('signal efficiency')
plt.legend()

# test roc auc
plt.subplot(1, 2, 2)
plt.title = 'MC ROC-AUC'
colors = cycle(['lightcoral', 'khaki', 'yellowgreen', 'lightblue', 'lightsteelblue'])
for ind, color in enumerate(colors):
    plt.plot(fpr_mc[ind], tpr_mc[ind], color=color,
             label='ROC curve of ' + mc_keys[ind] + ' (area = {0:0.4f})'
             ''.format(roc_auc_mc[ind]))
plt.plot(fpr_mc["micro"], tpr_mc["micro"], color='black', linestyle=':',
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc_mc["micro"]))
plt.xlabel('background efficiency')
plt.ylabel('signal efficiency')
plt.legend()

f.tight_layout()

plt.savefig('ROC_AUC_OvsR.pdf')


#adding distribution prob.
for key, prob in enumerate(mc_keys):
    df_data_test['prob_{0}'.format(key)] = df_data_pred_proba[:, prob]
    df_mc_test['prob_{0}'.format(key)] = df_mc_pred_proba[:, prob]

#color dictionary
col = {'electrons': 'blue', 'pi': 'orangered', 'kaons': 'red',
    'protons': 'green', 'deuterons': 'grey'}

#plot of keys
for prob_key in data_keys:
    fighist = plt.figure(figsize=[10,8])
    for key in data_keys:
        #plot histogram
        plt.hist(df_data_test.loc[df_data_test[key] == 1]['prob_{0}'.format(prob_key)], color = col[key],
        alpha =0.5, bins = 50, histtype='stepfilled', density=True,
        label = '{0}_train'.format(key), log=True)
        #error_bar
        hist, bins = np.histogram(df_mc_test.query('PDGcode == {0}'.format(pdg[key]))['prob_{0}'.format(prob_key)].values, bins = 50, density = True )
        scale = len(df_mc_test) / sum(hist)
        err = np.sqrt(hist * scale) / scale
        center = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(center, hist, yerr=err, fmt='o', c=col[key], label = '{0}_test'.format(key))
    plt.xlabel('probability')
    plt.ylabel('log(entries)')
    plt.xlim(0,1)
    plt.legend(loc='best')
    fighist.savefig('probability_distribution_of_{0}_and_OvsR.pdf'.format(prob_key))


plt.show()
