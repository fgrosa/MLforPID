import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import argparse
from itertools import product

def plotfinale2(file):
    """This function plot in a 2D histogram the data in the
     _MC files with correct PDGcode and not"""

    # pdgcode
    pdg = {'electrons': 11, 'pi': 211, 'kaons': 321, 'protons': 2212,
           'He3': 1000020030, 'triton': 1000010030, 'deuterons': 1000010020}

    # keys for dictionary
    keys = list(map(lambda x: x.split('_')[0], file))
    # dictionary for data
    data = dict(zip(keys, file))

    # create dataframe for dictionaries
    for k in data:
        data[k] = pd.read_parquet(data[k])

    # data dataframe
    data_pure = {}
    # correct pdg code
    for specie in data:
        new_df = data[specie].query('PDGcode == {0}'.format(pdg[specie]))
        data_pure.update({specie: new_df})

    # raw data plot
    ptraw = {}
    ptpure = {}

    for k in data:
        ptraw[k] = data[k].loc[:, 'p'].values
        ptpure[k] = data_pure[k].loc[:, 'p'].values

    #width for hist
    width3 = [0.3, 0.5, 0.75, 1, 1.5, 3, 5, 10]
    width2 = [0.3, 0.5, 0.75, 1, 1.5, 3, 5, 10]
    width = np.arange(0,6,0.1)

    #labels and colors
    color = ['royalblue', 'orangered', 'red', 'forestgreen', 'black', 'mediumslateblue', 'slategray']
    labels = {'electrons': 'electrons', 'pi': 'pions', 'kaons': 'kaons', 'protons': 'protons','He3': 'He', 'triton': 'triton', 'deuterons': 'deuterons'}

    #pure data vs raw percentage, dividing species in two groups for better visualization
    np.seterr(divide='ignore', invalid='ignore')
    xer, x = [], []

    for binposition in range(len(width3)-1):
        xer.append((width3[binposition+1]-width3[binposition])/2)
        x.append((width3[binposition]+width3[binposition+1])/2)

    _, axes = plt.subplots(2, 4, figsize=[12, 7])
    for ipad, (j, index) in enumerate(zip(labels, product([0,1], [0,1,2,3]))):
        dfraw = (pd.DataFrame(ptraw[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['sum']}))
        dfpure = (pd.DataFrame(ptpure[j], columns=['p']).assign(Bin=lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['sum']}))
        dfrawcount = (pd.DataFrame(ptraw[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['count']}))
        dfpurecount = (pd.DataFrame(ptpure[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['count']}))
        yer = np.divide(np.sqrt(np.asarray(dfpurecount)*(1 - np.divide(dfpurecount,dfrawcount))),dfrawcount)
        axes[index[0], index[1]].errorbar(x, np.asarray(np.divide(dfpure,dfraw)), xerr = xer , yerr = np.asarray(yer), c=color[ipad], fmt='o', capsize=3, elinewidth=0.5)
        axes[index[0], index[1]].set_xlabel('p (GeV/c)')
        axes[index[0], index[1]].set_ylabel('purity')
        axes[index[0], index[1]].set_xscale('log')
        axes[index[0], index[1]].set_ylim((0., 1.2))
        axes[index[0], index[1]].set_title(labels[j])
        axes.flat[-1].set_visible(False)
        plt.tight_layout()

    plt.savefig('plot_purity_MC.png')

    _, axes2 = plt.subplots(2, 4, figsize=[12, 7])
    for ipad, (j, index) in enumerate(zip(labels, product([0,1], [0,1,2,3]))):
        axes2[index[0], index[1]].hist(ptraw[j], bins = width, color = color[0], label = 'raw', edgecolor='white')
        axes2[index[0], index[1]].hist(ptpure[j], bins = width, color = color[2], label = 'pure', edgecolor='white', alpha = 0.7)
        axes2[index[0], index[1]].set_title('{0} sample'.format(labels[j]))
        axes2[index[0], index[1]].set_xlabel('p (GeV/c)')
        axes2[index[0], index[1]].set_ylabel('counts')
        axes2.flat[-1].set_visible(False)
        plt.tight_layout()
        plt.legend()

    plt.show()

# flags
PARSER = argparse.ArgumentParser(description='Arguments to pass')
PARSER.add_argument('dir', metavar='text', default='.',
                    help='flag to search in directory, remember put all paths!')
ARGS = PARSER.parse_args()


# files
files = [f for f in os.listdir(ARGS.dir) if 'MC.parquet.gzip' in f]
files.remove('kaons_fromTOF_MC.parquet.gzip')

# plot
os.chdir(ARGS.dir)
plotfinale2(files)