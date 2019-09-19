import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
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

#add category to dataframes
multi_column(data)

#training columns
training_columns = ['p','pTPC','ITSclsMap','dEdxITS','NclusterPIDTPC','dEdxTPC']

#training dataframe
training_df = pd.concat([data[keys].iloc[:5000] for keys in data],ignore_index = True)

#traing data with training columns
X_df = training_df[training_columns]
Y_df = training_df.filter(like='category', axis=1)

#parameters for grid search
max_depth=[2,3,5,8]
n_estimators=[100,200,500]
learning_rate=[0.1,0.2,0.3]

#all combination of parameter
parameters = product(max_depth,n_estimators,learning_rate)

#folding for cross validation
kf = KFold(n_splits=5, shuffle= True, random_state=42)

#list for dataframe of results and parameters
list_for_df = []

print('grid search started')

#manual grid search
for ipar,param in enumerate(parameters):
    for index in kf.split(training_df):
        # onevsrest classifier
        clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=param[0],
         n_estimators=param[1],learning_rate=param[2]))
        x_train = X_df.iloc[index[0]]
        y_train = Y_df.iloc[index[0]]
        x_test  = X_df.iloc[index[1]]
        y_test  = Y_df.iloc[index[1]]
        y_score = clf.fit(x_train,y_train).predict_proba(x_test)
        n_classes= range(len(data.keys()))
        fpr=dict()
        tpr=dict()
        roc_auc=dict()
        for cat in n_classes:
            fpr[cat], tpr[cat], _ = roc_curve(y_test.loc[:, 'category_{0}'.format(cat)].values, y_score[:, cat])
            roc_auc[cat] = auc(fpr[cat], tpr[cat]) 
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.loc[:, 'category_0':'category_{0}'.format(n_classes)].values.ravel(),y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #temporary list
        tmp_list = [ipar,param[0],param[1], param[2], roc_auc['micro']]
        list_for_df.append(tmp_list)

#list of columns for df
columns_df = ['index','n_estimator','max_depth','learning_rate','roc_auc_micro']
#dataframe of the results of grid search
df_results = pd.DataFrame(list_for_df,columns = columns_df)
#conversion to parquet
df_results.to_parquet('results_grid_search.parquet.gzip',compression='gzip')
print('dataframe of the results of grid search saved')
#plotting datas
df_results.plot(x='index', y='roc_auc_micro', kind ='line')
plt.show()