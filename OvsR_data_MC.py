import os
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from Add_category import multi_column
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

currentdir = os.getcwd()

# data_files necessary
data_files = [f for f in os.listdir(ARGS.data_dir) if (
    'data.parquet.gzip' in f and not f.startswith("._"))]
data_files.remove('kaons_fromTOF_data.parquet.gzip')
data_files.remove('He3_fromTOFTPC_data.parquet.gzip')
data_files.remove('triton_fromTOFTPC_data.parquet.gzip')
data_files.sort()
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

#order data keys and data files by name
data_keys.sort()
data_files.sort()

#load model saved
clf = pickle.load(open('OvsR_trained.pkl', 'rb'))

# MC_files necessary
os.chdir(currentdir)
mc_files = [f for f in os.listdir(ARGS.mc_dir) if (
    'MC.parquet.gzip' in f and not f.startswith("._"))]
mc_files.remove('kaons_fromTOF_MC.parquet.gzip')
mc_files.remove('He3_fromTOFTPC_MC.parquet.gzip')
mc_files.remove('triton_fromTOFTPC_MC.parquet.gzip')
mc_files.sort()
os.chdir(ARGS.mc_dir)

# pdgcode
pdg = {'electrons': 11, 'pi': 211, 'kaons': 321, 'protons': 2212,
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

#add category to dataframe
multi_column(df_mc)

#columns used to fit
target_columns = ['p', 'pTPC', 'ITSclsMap',
    'dEdxITS', 'NclusterPIDTPC', 'dEdxTPC']

#all dataframe for testing and roc auc
df_data_test = pd.concat([df_data[key].iloc[50000:]
                     for key in df_data], ignore_index=True)

#predict data based on the target columns
df_data_pred_proba = clf.predict_proba(df_data_test[target_columns])

#dataframe mc for test
df_mc_test = pd.concat([df_mc[key] for key in df_mc], ignore_index=True)

#predict mc based on target column
df_mc_pred_proba = clf.predict_proba(df_mc_test[target_columns])

#fpr ,tpr and roc_auc
fpr_data = dict()
tpr_data = dict()
fpr_mc = dict()
tpr_mc = dict()
roc_auc_data = dict()
roc_auc_mc = dict()

#calculation of roc_auc_data
roc_calculation(fpr_data, tpr_data, roc_auc_data,
    df_data_test[data_keys].values, df_data_pred_proba, len(df_data))
roc_calculation(fpr_mc, tpr_mc, roc_auc_mc,
    df_mc_test[mc_keys].values, df_mc_pred_proba, len(df_data))

#roc auc curve
# new figure for roc auc
f = plt.figure(figsize=[10, 5], constrained_layout=True)
# train roc auc
plt.subplot(1, 2, 1)
plt.title('Data', fontsize=25)
colors = ['silver', 'lightsteelblue', 'lightcoral', 'khaki', 'lightgreen']
for ind, color in enumerate (colors):
    plt.plot(fpr_data[ind], tpr_data[ind], color=color,
             label= data_keys[ind] + ' (area = {0:0.4f})'.format(roc_auc_data[ind]))
plt.plot(fpr_data["micro"], tpr_data["micro"], color='black', linestyle=':',
         label='micro-average (area = {0:0.4f})'.format(roc_auc_data["micro"]))
plt.xlabel('background efficiency',fontsize=20)
plt.ylabel('signal efficiency',fontsize=20)
plt.legend(loc ='lower right',fontsize=12)

# test roc auc
plt.subplot(1, 2, 2)
plt.title('MC',fontsize = 25)
colors = ['silver', 'lightsteelblue', 'lightcoral', 'khaki', 'lightgreen']
for ind, color in enumerate(colors):
    plt.plot(fpr_mc[ind], tpr_mc[ind], color=color,
             label=mc_keys[ind] + ' (area = {0:0.4f})'.format(roc_auc_mc[ind]))
plt.plot(fpr_mc["micro"], tpr_mc["micro"], color='black', linestyle=':',
         label='micro-average (area = {0:0.4f})'.format(roc_auc_mc["micro"]))
plt.xlabel('background efficiency',fontsize = 20)
plt.ylabel('signal efficiency',fontsize = 20)
plt.legend(loc ='lower right',fontsize = 12)

f.tight_layout()

plt.savefig('ROC_AUC_OvsR_data_mc.pdf')


#adding distribution prob.
for prob, key in enumerate(df_mc):
    df_data_test['prob_{0}'.format(key)] = df_data_pred_proba[:, prob]
    df_mc_test['prob_{0}'.format(key)] = df_mc_pred_proba[:, prob]

#color dictionary
col = {'electrons': 'blue', 'pi': 'orangered', 'kaons': 'red',
    'protons': 'green', 'deuterons': 'grey'}

#plot of keys
for prob_key in df_data:
    fighist = plt.figure(figsize=[10,8])
    for key in df_data:
        #plot histogram
        hist, bins, _ = plt.hist(df_data_test.loc[df_data_test[key] == 1]['prob_{0}'.format(prob_key)], color = col[key],
        alpha = 1, bins = 100, histtype='step', density=True, label = '{0}_data'.format(key), log=True)
        center = (bins[:-1] + bins[1:]) / 2
        #plt.fill_between(center, [1.e-4 for b in range(len(center))], hist, color = col[key], alpha=0.25)
        plt.ylim(1.e-4,2.e2)
        #error_bar
        hist, bins = np.histogram(df_mc_test.loc[df_mc_test[key] == 1]['prob_{0}'.format(prob_key)].values, bins = 100, density = True )
        scale = len(df_mc_test) / sum(hist)
        err = np.sqrt(hist * scale) / scale
        plt.errorbar(center, hist, yerr=err, fmt='o', c=col[key], label = '{0}_MC'.format(key))
    plt.xlabel('probability to be {0}'.format(prob_key), fontsize=20)
    plt.ylabel('entries', fontsize=20)
    plt.xlim(0,1)
    plt.legend(loc='best',fontsize=14)
    fighist.savefig('probability_distribution_of_{0}_and_OvsR.pdf'.format(prob_key))

plt.show()
