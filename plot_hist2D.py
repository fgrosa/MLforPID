import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import argparse

# functions MC


def hist2D_MC(file):
    """This function plot in a 2D histogram the data in the
     _MC files with correct PDGcode and not"""

    # pdgcode
    pdg = {'e': 11, 'pi': 211, 'kaons': 321, 'p': 2212,
           'He3': 1000020030, 'triton': 1000010030, 'deuterons': 1000010020}

    # cmap
    cmap = {'e': 'Blues', 'pi': 'Oranges', 'kaons': 'Reds', 'p': 'Greens',
            'He3': 'bone', 'triton': 'Purples', 'deuterons': 'Greys'}

    # keys for dictionary
    keys = list(map(lambda x: x.split('_')[0], file))
    # dictionary for data
    data = dict(zip(keys, file))

    # create dataframe for dictionaries
    for k in data:
        data[k] = pd.read_parquet(data[k]).iloc[:1000]

    # data dataframe
    data_pure = {}
    # correct pdg code
    for specie in data:
        new_df = data[specie].query('PDGcode == {0}'.format(pdg[specie]))
        data_pure.update({specie: new_df})

    # raw data plot
    dEitsraw = {}
    ptraw = {}
    dEtpcraw = {}
    ptpcraw = {}
    # data data plot
    dEits = {}
    ptpure = {}
    dEtpc = {}
    ptpc = {}

    # raw data for plot
    for k in data:
        ptraw[k] = data[k].loc[:, 'p'].values
        ptpcraw[k] = data[k].loc[:, 'pTPC'].values
        dEitsraw[k] = data[k].loc[:, 'dEdxITS'].values
        dEtpcraw[k] = data[k].loc[:, 'dEdxTPC'].values
    # data data for plot
    for k in data:
        ptpure[k] = data_pure[k].loc[:, 'p'].values
        ptpc[k] = data_pure[k].loc[:, 'pTPC'].values
        dEits[k] = data_pure[k].loc[:, 'dEdxITS'].values
        dEtpc[k] = data_pure[k].loc[:, 'dEdxTPC'].values

    # plot hist2D
    labels = {'e': 'electons', 'pi': 'pions', 'kaons': 'kaons', 'p': 'protons',
              'He3': 'He', 'triton': 'triton', 'deuterons': 'deuterons'}

    # subplot
    f, ax = plt.subplots(nrows=2, ncols=2)

    # ITS PURE
    ax[0, 0].set_title('ITS PURE')
    for k in data:
        ax[0, 0].hist2d(ptpure[k], dEits[k], cmap=plt.get_cmap(cmap[k]), alpha=0.5,
                        range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(400, 200), label=labels[k], norm=LogNorm(1.e-1, 1.e2))
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_ylabel('dE/dx')
    ax[0, 0].set_xlabel('p (GeV/c)')
    ax[0, 0].set_xlim((1e-1, 2e+1))
    ax[0, 0].legend()

    # ITS RAW
    ax[1, 0].set_title('ITS RAW')
    for k in data:
        ax[1, 0].hist2d(ptraw[k], dEitsraw[k], cmap=plt.get_cmap(cmap[k]), alpha=0.5,
                        range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(400, 200), label=labels[k], norm=LogNorm(1.e-1, 1.e2))
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_ylabel('dE/dx')
    ax[1, 0].set_xlabel('p (GeV/c)')
    ax[1, 0].set_xlim((1e-1, 2e+1))
    ax[1, 0].legend()

    # TPC PURE
    ax[0, 1].set_title('TPC PURE')
    for k in data:
        ax[0, 1].hist2d(ptpc[k], dEtpc[k], cmap=plt.get_cmap(cmap[k]), alpha=0.5,
                        range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(400, 200), label=labels[k], norm=LogNorm(1.e-1, 1.e2))
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_ylabel('dE/dx')
    ax[0, 1].set_xlabel('p (GeV/c)')
    ax[0, 1].set_xlim((1e-1, 2e+1))
    ax[0, 1].legend()

    # TPC RAW
    ax[1, 1].set_title('TPC RAW')
    for k in data:
        ax[1, 1].hist2d(ptpcraw[k], dEtpcraw[k],  cmap=plt.get_cmap(cmap[k]), alpha=0.5,
                        range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(400, 200), label=labels[k], norm=LogNorm(1.e-1, 1.e2))
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_ylabel('dE/dx')
    ax[1, 1].set_xlabel('p (GeV/c)')
    ax[1, 1].set_xlim((1e-1, 2e+1))
    ax[1, 1].legend()
    # add some space
    plt.tight_layout()

    plt.savefig('plot_hist2D_MC.pdf')
    plt.show()

# functions data


def hist2D_data(file):
    """This function plot in a 2D histogram the data in the
     _data files"""

    # cmap
    cmap = {'e': 'Blues', 'pi': 'Oranges', 'kaons': 'Reds', 'p': 'Greens',
            'He3': 'PuBu', 'triton': 'Purples', 'deuterons': 'Greys'}

    # keys for dictionary
    keys = list(map(lambda x: x.split('_')[0], file))
    # dictionary for data
    data = dict(zip(keys, file))

    # create dataframe for dictionaries
    for k, df in zip(data.keys(), data.values()):
        data[k] = pd.read_parquet(df).iloc[:1000]

    # data plot
    dEits = {}
    pt = {}
    dEtpc = {}
    ptpc = {}

    # data for plot
    for k in data:
        pt[k] = data[k].loc[:, 'p'].values
        ptpc[k] = data[k].loc[:, 'pTPC'].values
        dEits[k] = data[k].loc[:, 'dEdxITS'].values
        dEtpc[k] = data[k].loc[:, 'dEdxTPC'].values

    # plot hist2D
    labels = {'e': 'electons', 'pi': 'pions', 'kaons': 'kaons', 'p': 'protons',
              'He3': 'He', 'triton': 'triton', 'deuterons': 'deuterons'}

    # subplot
    f, ax = plt.subplots(nrows=1, ncols=2)
    plt.grid(True)

    # ITS
    ax[0].set_title('ITS')
    for k in data:
        ax[0].hist2d(pt[k], dEits[k], cmap=plt.get_cmap(cmap[k]), alpha=0.5,
                     range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(400, 200), label=labels[k], norm=LogNorm(1.e-1, 1.e2))
    ax[0].set_xscale('log')
    ax[0].set_ylabel('dE/dx')
    ax[0].set_xlabel('p (GeV/c)')
    ax[0].set_xlim((1e-1, 2e+1))
    ax[0].legend()

    # TPC
    ax[1].set_title('TPC')
    for k in data:
        ax[1].hist2d(ptpc[k], dEtpc[k],  cmap=plt.get_cmap(cmap[k]), alpha=0.5,
                     range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(400, 200), label=labels[k], norm=LogNorm(1.e-1, 1.e2))  # , bins=(200, 200)
    ax[1].set_xscale('log')
    ax[1].set_ylabel('dE/dx')
    ax[1].set_xlabel('p (GeV/c)')
    ax[1].set_xlim((1e-1, 2e+1))
    ax[1].legend()
    # add some space
    plt.tight_layout()

    plt.savefig('plot_hist2D_data.pdf')
    plt.show()


# flags
PARSER = argparse.ArgumentParser(description='Arguments to pass')
PARSER.add_argument('dir', metavar='text', default='.',
                    help='flag to search in directory, remember put all paths!')
PARSER.add_argument('--data', action='store_true',
                    help='flag to search and visualize data file')
PARSER.add_argument('--mc', action='store_true',
                    help='flag to search and visualize Monte Carlo file')
ARGS = PARSER.parse_args()

if not ARGS.data and not ARGS.mc:
    print('You need to specificy if mc or data files')
    exit()

if ARGS.mc:
    # files
    files = [f for f in os.listdir(ARGS.dir) if 'MC.parquet.gzip' in f]
    files.remove('kaons_fromTOF_MC.parquet.gzip')
    os.chdir(ARGS.dir)
    # plot
    
    hist2D_MC(files)

elif ARGS.data:
    # files
    files = [f for f in os.listdir(ARGS.dir) if '_data.parquet.gzip' in f]
    files.remove('kaons_fromTOF_data.parquet.gzip')
    os.chdir(ARGS.dir)
    # plot
    
    hist2D_data(files)
