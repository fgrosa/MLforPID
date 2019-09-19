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
    pdg = {'e': 11, 'pi': 211, 'kaons': 321, 'p': 2212,
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
    color = ['blue','yellow','green','red']
    color2 = ['grey','black', 'purple']
    nomi = ['electrons', 'pions', 'kaons', 'protons','He3','tritons','deuterons','pure','raw']
    labels = {'e': 'electrons', 'pi': 'pions', 'kaons': 'kaons', 'p': 'protons'}
    labels2 = {'He3': 'He', 'triton': 'triton', 'deuterons': 'deuterons'}
   
    #pure data vs raw percentage, dividing species in two groups for better visualization
    np.seterr(divide='ignore', invalid='ignore')
    xer, x = [], []

    for binposition in range(len(width3)-1):
        xer.append((width3[binposition+1]-width3[binposition])/2)
        x.append((width3[binposition]+width3[binposition+1])/2)

    fig1, axes = plt.subplots(2, 2, figsize=[12, 7])
    for ipad, (j, index) in enumerate(zip(labels, product([0,1], [0,1]))):
        # plt.figure(3)
        # plt.subplot(2,2,z+1)
        dfraw = (pd.DataFrame(ptraw[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['sum']}))
        dfpure = (pd.DataFrame(ptpure[j], columns=['p']).assign(Bin=lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['sum']}))
        dfrawcount = (pd.DataFrame(ptraw[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['count']}))
        dfpurecount = (pd.DataFrame(ptpure[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['count']}))
        yer = np.divide(np.sqrt(np.asarray(dfpurecount)*(1 - np.divide(dfpurecount,dfrawcount))),dfrawcount)
        # axes[index[0], index[1]].scatter(width3, np.divide(dfpure,dfraw), marker='o', c=color[ipad])
        # plt.errorbar(np.log(width3), np.asarray(np.divide(dfpure,dfraw)), xerr = xer , yerr = np.asarray(yer), fmt='.k', capsize=3, elinewidth=0.5)
        axes[index[0], index[1]].errorbar(x, np.asarray(np.divide(dfpure,dfraw)), xerr = xer , yerr = np.asarray(yer), c=color[ipad], fmt='o', ecolor='k', capsize=3, elinewidth=0.5, marker='o')
        axes[index[0], index[1]].set_xlabel('p (GeV/c)')
        axes[index[0], index[1]].set_ylabel('purity')
        axes[index[0], index[1]].set_xscale('log')
        axes[index[0], index[1]].set_ylim((0., 1.2))
        axes[index[0], index[1]].set_title(labels[j])
        plt.tight_layout()
    
    fig2, axes2 = plt.subplots(2, 2, figsize=[12, 7])
    for ipad, (j, index) in enumerate(zip(labels2, product([0,1], [0,1]))):
        dfraw = (pd.DataFrame(ptraw[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['sum']}))
        dfpure = (pd.DataFrame(ptpure[j], columns=['p']).assign(Bin=lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['sum']}))
        dfrawcount = (pd.DataFrame(ptraw[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['count']}))
        dfpurecount = (pd.DataFrame(ptpure[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['count']}))
        yer = np.divide(np.sqrt(np.asarray(dfpurecount)*(1 - np.divide(dfpurecount,dfrawcount))),dfrawcount)
        axes2[index[0], index[1]].errorbar(x, np.asarray(np.divide(dfpure,dfraw)), xerr = xer , yerr = np.asarray(yer), c=color2[ipad], fmt='o', ecolor='k', capsize=3, elinewidth=0.5, marker='o')
        axes2[index[0], index[1]].set_xlabel('p (GeV/c)')
        axes2[index[0], index[1]].set_ylabel('purity')
        axes2[index[0], index[1]].set_xscale('log')
        axes2[index[0], index[1]].set_ylim((0., 1.2))
        axes2[index[0], index[1]].set_title(labels2[j])
        axes2.flat[-1].set_visible(False)
        plt.tight_layout()

    plt.figure(3)
    #plot hist electrons
    plt.subplot(2,2,1)    
    plt.hist(ptraw['e'], bins = width, color = color[0], label = nomi[8], edgecolor='white')
    plt.hist(ptpure['e'], bins = width, color = color[3], label = nomi[7], edgecolor='white', alpha = 0.7)
    plt.title('electron sample')
    plt.xlabel('P')
    plt.ylabel('conteggio')
    plt.legend()

    #plot hist pions
    plt.subplot(2,2,2)
    plt.hist(ptraw['pi'],bins = width,color = color[0], label = nomi [8],edgecolor='white')
    plt.hist(ptpure['pi'],bins = width,color = color[3], label = nomi [7],edgecolor='white', alpha = 0.7)
    plt.title('pions sample')
    plt.xlabel('P')
    plt.ylabel('conteggio')
    plt.legend()

    #plot hist kaons
    plt.subplot(2,2,3)
    plt.hist(ptraw['kaons'],bins = width,color = color[0], label = nomi [8],edgecolor='white')
    plt.hist(ptpure['kaons'],bins = width,color = color[3], label = nomi [7],edgecolor='white', alpha = 0.7)
    plt.title('kaons sample')
    plt.xlabel('P')
    plt.ylabel('conteggio')
    plt.legend()

    #plot hist protons
    plt.subplot(2,2,4)
    plt.hist(ptraw['p'],bins = width,color = color[0], label = nomi [8],edgecolor='white')
    plt.hist(ptpure['p'],bins = width,color = color[3], label = nomi [7],edgecolor='white', alpha = 0.7)
    plt.title('protons sample')
    plt.xlabel('P')
    plt.ylabel('conteggio')
    plt.legend()

    plt.tight_layout()

    plt.figure(4)

    #plot hist deuterons
    plt.subplot(2,2,1)
    plt.hist(ptraw['deuterons'],bins = width,color = color[0], label = nomi [8],edgecolor='white')
    plt.hist(ptpure['deuterons'],bins = width,color = color[3], label = nomi [7],edgecolor='white', alpha = 0.7)
    plt.title('deuterons sample')
    plt.xlabel('P')
    plt.ylabel('conteggio')
    plt.legend()

    #plot hist tritons
    plt.subplot(2,2,2)
    plt.hist(ptraw['triton'],bins = width,color = color[0], label = nomi [8],edgecolor='white')
    plt.hist(ptpure['triton'],bins = width,color = color[3], label = nomi [7],edgecolor='white', alpha = 0.7)
    plt.title('tritons sample')
    plt.xlabel('P')
    plt.ylabel('conteggio')
    plt.legend()

    #plot hist He3
    plt.subplot(2,2,3)
    plt.hist(ptraw['He3'],bins = width,color = color[0], label = nomi [8],edgecolor='white')
    plt.hist(ptpure['He3'],bins = width,color = color[3], label = nomi [7],edgecolor='white', alpha = 0.7)
    plt.title('He3 sample')
    plt.xlabel('P')
    plt.ylabel('conteggio')
    plt.legend()

    plt.tight_layout()

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
