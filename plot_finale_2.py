import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import argparse


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
        data[k] = pd.read_parquet(data[k]).iloc[:1000]

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
    width2 = [0, 0.3, 0.5, 0.75, 1, 1.5, 3, 5, 10]
    width = np.arange(0,6,0.1)

    #labels and colors
    color = ['blue','yellow','green','red','grey','black', 'purple']
    nomi = ['electrons', 'pions', 'kaons', 'protons','He3','tritons','deuterons','pure','raw']
    labels = {'e': 'electrons', 'pi': 'pions', 'kaons': 'kaons', 'p': 'protons',
              'He3': 'He', 'triton': 'triton', 'deuterons': 'deuterons'}

    #pure data vs raw raw percentage
    np.seterr(divide='ignore', invalid='ignore')
    xer = []

    for binposition in range(len(width3)):
        xer.append((width3[binposition]-width3[binposition-1])/2)
        if xer[binposition] < 0:
            xer[binposition] = 0

    z=0
    for j in labels:
        plt.figure(3)
        plt.subplot(4,2,z+1)
        dfraw = (pd.DataFrame(ptraw[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['sum']}))
        dfpure = (pd.DataFrame(ptpure[j], columns=['p']).assign(Bin=lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['sum']}))
        dfrawcount = (pd.DataFrame(ptraw[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['count']}))
        dfpurecount = (pd.DataFrame(ptpure[j], columns=['p']).assign(Bin = lambda x: pd.cut(x.p, bins=width2)).groupby(['Bin']).agg({'p': ['count']}))
        yer = np.divide(np.sqrt(np.asarray(dfpurecount)*(1 - np.divide(dfpurecount,dfrawcount))),dfrawcount)
        plt.scatter(width3, np.asarray(np.divide(dfpure,dfraw)), marker='o', c=color[z])
        plt.errorbar(width3, np.asarray(np.divide(dfpure,dfraw)), xerr = xer , yerr = np.asarray(yer), fmt='.k', capsize=3, elinewidth=0.5)    
        plt.title(nomi[z])
        plt.axis([0,13,0,1.3])    
        plt.xlabel('P')
        plt.ylabel('% pure')
        plt.tight_layout()
        z = z+1

    plt.figure(1)
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

    plt.figure(2)

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

    # plot
    os.chdir(ARGS.dir)
    plotfinale2(files)

elif ARGS.data:
    # files
    files = [f for f in os.listdir(ARGS.dir) if '_data.parquet.gzip' in f]
    files.remove('kaons_fromTOF_data.parquet.gzip')

    # plot
    os.chdir(ARGS.dir)
    plotfinale2(files)