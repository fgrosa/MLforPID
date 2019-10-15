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
    pdg = {'electrons': 11, 'pi': 211, 'kaons': 321, 'protons': 2212,
           'He3': 1000020030, 'triton': 1000010030, 'deuterons': 1000010020}

    # cmap
    cmap = {'electrons': 'Blues', 'pi': 'Oranges', 'kaons': 'Reds', 'protons': 'Greens',
            'He3': 'bone', 'triton': 'Purples', 'deuterons': 'Greys'}
    print(cmap)

    # keys for dictionary
    keys = list(map(lambda x: x.split('_')[0], file))
    # dictionary for data
    data = dict(zip(keys, file))

    # create dataframe for dictionaries
    for key in data:
        data[key] = pd.read_parquet(data[key])
        header = data[key].select_dtypes(include=[np.float64]).columns
        # change dtype of column with float64 to float16
        data[key][header] = data[key][header].astype('float16')

    # data dataframe
    data_pure = {}
    # correct pdg code
    for specie in data:
        new_df = data[specie].query('PDGcode == {0}'.format(pdg[specie]))
        data_pure.update({specie: new_df})

    # plot hist2D
    labels = {'electrons': 'electons', 'pi': 'pions', 'kaons': 'kaons', 'protons': 'protons',
              'He3': 'He', 'triton': 'triton', 'deuterons': 'deuterons'}

    # subplot
    f, ax = plt.subplots(nrows=2, ncols=2)

    # ITS PURE
    ax[0, 0].set_title('ITS PURE')
    for key in data:
        ax[0, 0].hist2d(data_pure[key]['p'].values, data_pure[key]['dEdxITS'].values, cmap=plt.get_cmap(cmap[key]), alpha=0.5,
                        range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(1000, 1000), label=labels[key], norm=LogNorm(1.e-1, 1.e2))
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_ylabel('dE/dx')
    ax[0, 0].set_xlabel('p (GeV/c)')
    ax[0, 0].set_xlim((1e-1, 2e+1))
    

    # ITS RAW
    ax[1, 0].set_title('ITS RAW')
    for key in data:
        ax[1, 0].hist2d(data[key]['p'].values, data[key]['dEdxITS'].values, cmap=plt.get_cmap(cmap[key]), alpha=0.5,
                        range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(1000, 1000), label=labels[key], norm=LogNorm(1.e-1, 1.e2))
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_ylabel('dE/dx')
    ax[1, 0].set_xlabel('p (GeV/c)')
    ax[1, 0].set_xlim((1e-1, 2e+1))
   
    # TPC PURE
    ax[0, 1].set_title('TPC PURE')
    for key in data:
        ax[0, 1].hist2d(data_pure[key]['pTPC'].values,data_pure[key]['dEdxTPC'].values, cmap=plt.get_cmap(cmap[key]), alpha=0.5,
                        range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(1000,1000), label=labels[key], norm=LogNorm(1.e-1, 1.e2))
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_ylabel('dE/dx')
    ax[0, 1].set_xlabel('p (GeV/c)')
    ax[0, 1].set_xlim((1e-1, 2e+1))
    
    # TPC RAW
    ax[1, 1].set_title('TPC RAW')
    for key in data:
        ax[1, 1].hist2d(data[key]['pTPC'].values, data[key]['dEdxTPC'].values,  cmap=plt.get_cmap(cmap[key]), alpha=0.5,
                        range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(1000,1000), label=labels[key], norm=LogNorm(1.e-1, 1.e2))
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_ylabel('dE/dx')
    ax[1, 1].set_xlabel('p (GeV/c)')
    ax[1, 1].set_xlim((1e-1, 2e+1))
    
    # add some space
    plt.tight_layout()

    plt.savefig('plot_hist2D_MC.png')
    plt.show()

# functions data
def hist2D_data(file):
    """This function plot in a 2D histogram the data in the
     _data files"""

    # cmap
    cmap = {'electrons': 'Blues', 'pi': 'Oranges', 'kaons': 'Reds', 'protons': 'Greens',
            'He3': 'PuBu', 'triton': 'Purples', 'deuterons': 'Greys'}
    print(cmap)

    # keys for dictionary
    keys = list(map(lambda x: x.split('_')[0], file))
    # dictionary for data
    data = dict(zip(keys, file))

    # create dataframe for dictionaries
    for key in data:
        data[key] = pd.read_parquet(data[key])
        header = data[key].select_dtypes(include=[np.float64]).columns
        # change dtype of column with float64 to float16
        data[key][header] = data[key][header].astype('float16')

    # plot hist2D
    labels = {'electrons': 'electrons', 'pi': 'pions', 'kaons': 'kaons', 'protons': 'protons',
              'He3': 'He', 'triton': 'triton', 'deuterons': 'deuterons'}

    # subplot
    f, ax = plt.subplots(nrows=1, ncols=2)
    plt.grid(True)

    # ITS
    ax[0].set_title('ITS')
    for key in data:
        ax[0].hist2d(data[key]['p'].values, data[key]['dEdxITS'].values, cmap=plt.get_cmap(cmap[key]), alpha=0.5,
                     range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(1000,1000), label=labels[key], norm=LogNorm(1.e-1, 1.e2))
    ax[0].set_xscale('log')
    ax[0].set_ylabel('dE/dx')
    ax[0].set_xlabel('p (GeV/c)')
    ax[0].set_xlim((1e-1, 2e+1))
   
    # TPC
    ax[1].set_title('TPC')
    for key in data:
        ax[1].hist2d(data[key]['pTPC'].values, data[key]['dEdxTPC'].values,  cmap=plt.get_cmap(cmap[key]), alpha=0.5,
                     range=np.array([(1.e-1, 2e1), (0, 1000)]), bins=(1000,1000), label=labels[key], norm=LogNorm(1.e-1, 1.e2))  # , bins=(200, 200)
    ax[1].set_xscale('log')
    ax[1].set_ylabel('dE/dx')
    ax[1].set_xlabel('p (GeV/c)')
    ax[1].set_xlim((1e-1, 2e+1))
    
    # add some space
    plt.tight_layout()
    plt.savefig('plot_hist2D_data.png')
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
    files = [f for f in os.listdir(ARGS.dir) if ('MC.parquet.gzip' in f and not f.startswith('._'))]
    files.remove('kaons_fromTOF_MC.parquet.gzip')
    files.sort()
    os.chdir(ARGS.dir)
    # plot
    hist2D_MC(files)

elif ARGS.data:
    # files
    files = [f for f in os.listdir(ARGS.dir) if ('_data.parquet.gzip' in f and not f.startswith('._'))]
    files.remove('kaons_fromTOF_data.parquet.gzip')
    files.sort()
    os.chdir(ARGS.dir)
    # plot
    hist2D_data(files)