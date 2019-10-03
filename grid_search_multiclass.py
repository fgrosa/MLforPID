import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from Add_category import single_column
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

for key in data:
    data[key] = pd.read_parquet(data[key])

print('adding category to dataframes')

# add category to dataframes
single_column(data)

# training columns
training_columns = ['p', 'pTPC', 'ITSclsMap',
                    'dEdxITS', 'NclusterPIDTPC', 'dEdxTPC']

# training dataframe
training_df = pd.concat([data[key].iloc[:5000]
                         for key in data], ignore_index=True)

# traing data with training columns
X_df = training_df[training_columns]
Y_df = training_df['category']

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

#creating dataframe for parameters values
paramsvalue = []

print('grid search started')

#manual grid search
for n, param in enumerate(parameters):
    scores = []
    for index in kf.split(training_df):
        # onevsrest classifier
        x_train = X_df.iloc[index[0]]
        y_train = Y_df.iloc[index[0]]
        x_test = X_df.iloc[index[1]]
        y_test = Y_df.iloc[index[1]]
        clf = XGBClassifier(n_jobs=-1, max_depth=param[0], n_estimators=param[1], learning_rate=param[2])
        clf.fit(x_train, y_train)
        y_score = clf.predict_proba(x_test)
        y_test_multi = label_binarize(y_test, classes=range(len(keys)))
        score = roc_auc_score(y_test_multi, y_score, average = 'micro')
        scores.append(score)
    mean = np.mean(np.array(scores))
    std  = np.std(np.array(scores))
    paramsvalue.append([n,param[0],param[1],param[2],mean,std])

# list of columns for df
columns_df = ['index', 'max_depth', 'n_estimator',
               'learning_rate', 'mean_roc_auc', 'root_mean_square']
# dataframe of the results of grid search
df_results = pd.DataFrame(paramsvalue, columns=columns_df)

# conversion to parquet
df_results.to_parquet('results_grid_search_multiclass.parquet.gzip', compression='gzip')
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
