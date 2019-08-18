#!/usr/bin/env python3
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from interpret.data import ClassHistogram
import matplotlib.pyplot as plt
from plotting_utils import make_meshgrid, plot_contours

#generate random distributions for 2 cathegories
x_A, y_A = (np.random.normal(0, 1, 5000) for ivar in range(2))
x_B, y_B = (np.random.normal(-2, 1, 5000) for ivar in range(2))

#define pandas df for each cathegory
df_A = pd.DataFrame({'x': x_A, 'y': y_A}, columns=['x', 'y'])
df_B = pd.DataFrame({'x': x_B, 'y': y_B}, columns=['x', 'y'])
df_A['category'] = 0
df_B['category'] = 1

#define training df (first 500 elements of each cathegory)
training_columns = ['x','y']
training_df = pd.concat([df_A.iloc[:500], df_B.iloc[:500]], ignore_index=True, sort=True) 

#define test df (second 500 elements of each cathegory)
test_df = pd.concat([df_A.iloc[500:], df_B.iloc[500:]], ignore_index=True, sort=True) 

ebm_clf = ExplainableBoostingClassifier()
ebm_clf.fit(training_df[training_columns], training_df['category'])

probabilities = ebm_clf.predict_proba(test_df[training_columns])
ebm_global = ebm_clf.explain_global()
show(ebm_global)

for prob in range(2):
  test_df['prob_{0}'.format(prob)] = probabilities[:,prob]

figcontur = plt.figure(figsize=(18,7.5))
contourax = figcontur.add_subplot(111)
xx, yy = make_meshgrid(test_df['x'], test_df['y'])
plot_contours(contourax, ebm_clf, xx, yy,cmap='RdYlBu', alpha=0.8)
contourax.scatter(test_df.x, test_df.y, c=test_df['category'], cmap='RdYlBu', s=20, edgecolors='k')
contourax.set_xlim(xx.min(), xx.max())
contourax.set_ylim(yy.min(), yy.max())
contourax.set_xlabel('x')
contourax.set_ylabel('y')
contourax.set_xticks(())
contourax.set_yticks(())

fighistos = plt.figure(figsize=(18,7.5))
for icat in range(2):
    sub = fighistos.add_subplot(1, 2, icat+1)
    plt.hist(test_df.loc[test_df['category'] == 0]['prob_{0}'.format(icat)], bins=50, histtype='step', density=1, label='A')
    plt.hist(test_df.loc[test_df['category'] == 1]['prob_{0}'.format(icat)], bins=50, histtype='step', density=1, label='B')
    plt.xlabel('prob_{0}'.format(icat))
    plt.ylabel('entries')
    plt.xlim(0,1)
    plt.yscale('log')
    plt.grid()
    plt.legend(loc='best')

plt.show()
