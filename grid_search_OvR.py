import numpy as np
import pandas as pd
import os
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,accuracy_score,classification_report
from Add_category import add_category


# files
files = [f for f in os.listdir('.') if 'MC.parquet.gzip' in f]
files.remove('kaons_fromTOF_MC.parquet.gzip')
os.chdir('.')

# keys for dictionary
keys = list(map(lambda x: x.split('_')[0], files))
# dictionary for data
data = dict(zip(keys, files))

# create dataframe for dictionaries
for keys in data:
    data[keys] = pd.read_parquet(data[keys])

#add category to dataframes
add_category(data)

#training columns
training_columns = ['p','pTPC','ITSclmap','dEdxITS','NclusterPIDTPC','dEdxTPC']

#training dataframe
training_df = pd.concat([data[keys].iloc[:5000] for keys in data],ignore_index = True)

#test dataframe
test_df = pd.concat([data[keys].iloc[5000:] for keys in data],ignore_index = True)

#onevsrest classifier
classifier = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))

#parameters for grid search
parameters = {
    'max_depth':[5,10,15],
    'n_estimators':[100,500,1000],
}

#function scoring
scores = ['accuracy_score','f1_score','precision','recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score + '\n')
    #grid search definition
    grid_search = GridSearchCV(classifier,param_grid = parameters,
        n_jobs=-1,scoring= '%s_macro' % score,cv=5)
    grid_search.fit(training_df[training_columns],training_df['category'])
    #print best parameters
    print("Best parameters set found on development set:\n ")
    print(grid_search.best_params_)
    #print scores
    print("Grid scores on development set:\n ")
    #variables
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']
    for mean, std, params in zip(means, stds, params):
        print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params) + '\n')
