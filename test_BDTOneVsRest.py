#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# generate random distributions for 4 cathegories
x_A, y_A = (np.random.normal(0, 1, 5000) for ivar in range(2))
x_B, y_B = (np.random.normal(2, 1, 5000) for ivar in range(2))
x_C, y_C = (np.random.normal(10*(-1*ivar), 2, 5000) for ivar in range(2))
x_D, y_D = (np.random.normal(-7, 4, 5000) for ivar in range(2))

# define pandas df for each cathegory
df_A = pd.DataFrame({'x': x_A, 'y': y_A}, columns=['x', 'y'])
df_B = pd.DataFrame({'x': x_B, 'y': y_B}, columns=['x', 'y'])
df_C = pd.DataFrame({'x': x_C, 'y': y_C}, columns=['x', 'y'])
df_D = pd.DataFrame({'x': x_D, 'y': y_D}, columns=['x', 'y'])
df_A['category'] = 0
df_B['category'] = 1
df_C['category'] = 2
df_D['category'] = 3

# define training df (first 500 elements of each cathegory)
training_columns = ['x', 'y']
training_df = pd.concat([df_A.iloc[:500], df_B.iloc[:500],
                         df_C.iloc[:500], df_D.iloc[:500]], ignore_index=True, sort=True)

# define test df (second 500 elements of each cathegory)
test_df = pd.concat([df_A.iloc[500:], df_B.iloc[500:],
                     df_C.iloc[500:], df_D.iloc[500:]], ignore_index=True, sort=True)

# define classifier
clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))

# do the training
clf.fit(training_df[training_columns], training_df['category'])

# predict probabilities for test set
probabilities = clf.predict_proba(test_df[training_columns])

for prob in range(4):
    test_df['prob_{0}'.format(prob)] = probabilities[:, prob]

# plots
fighistos = plt.figure(figsize=(18, 7.5))
for icat in range(4):
    sub = fighistos.add_subplot(2, 2, icat+1)
    plt.hist(test_df.loc[test_df['category'] == 0]['prob_{0}'.format(
        icat)], bins=50, histtype='step', density=1, label='A')
    plt.hist(test_df.loc[test_df['category'] == 1]['prob_{0}'.format(
        icat)], bins=50, histtype='step', density=1, label='B')
    plt.hist(test_df.loc[test_df['category'] == 2]['prob_{0}'.format(
        icat)], bins=50, histtype='step', density=1, label='C')
    plt.hist(test_df.loc[test_df['category'] == 3]['prob_{0}'.format(
        icat)], bins=50, histtype='step', density=1, label='D')
    plt.xlabel('prob_{0}'.format(icat))
    plt.ylabel('entries')
    plt.xlim(0, 1)
    plt.yscale('log')
    plt.grid()
    plt.legend(loc='best')

plt.show()
