import os
import numba
import uproot
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import argparse

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--maxentries', type=int, help='max number of entries')
parser.add_argument('--data', action='store_true', help='flag to add "data" suffix')
parser.add_argument('--mc', action='store_true', help='flag to add "MC" suffix')
parser.add_argument('inputfile', metavar='text', default='AnalysisResults.root', help='input root file name')
parser.add_argument('inputdir', metavar='text', default='PWGHF_D2H_SystNsigmaPID_pp_kINT7', help='TDirectorFile name in input root file')
parser.add_argument('outdir', metavar='text', default='.', help='output directory')
args = parser.parse_args()

def scalevars(df):
    df.pT /= 1000
    df.eta /= 1000
    df.phi /= 1000
    df.p /= 1000
    df.pTPC /= 1000
    df.pTOF /= 1000
    df.pHMPID /= 1000
    df.dEdxITS /= 100
    df.dEdxTPC /= 100
    df.ToF /= 10
    df.TrackLength /= 10
    df.StartTimeRes /= 100
    df.HMPIDsig /= 100
    df.HMPIDocc /= 100

@numba.njit
def selectbiton(array_tag, bits):
    is_tagged = []
    for track_type in array_tag:
        hastag = 0
        for bit in bits:
            if (track_type >> bit) & 0x1:
                hastag = 1
                break
        if hastag == 1:
            is_tagged.append(True)
        else:
            is_tagged.append(False)
    return is_tagged

def filter_bit_df(df, activatedbit):
    array_tag = df.loc[:, 'tag'].values.astype('int')
    res_on = pd.Series([True]*len(array_tag))

    bitmapon = selectbiton(array_tag, activatedbit)
    res_on = pd.Series(bitmapon)
    
    df_sel = df[res_on.values]
    return df_sel

if '/home' not in args.outdir or '/Users' not in args.outdir or '$HOME' not in args.outdir:
    args.outdir = os.getcwd()+'/'+args.outdir

if not os.path.isdir(args.outdir):
    print('creating output directory {0}'.format(args.outdir))
    os.mkdir(args.outdir)

tree = uproot.open(args.inputfile)['{0}/fPIDtree'.format(args.inputdir)]
if args.maxentries:
    df = tree.pandas.df(entrystop=args.maxentries)
else:
    df = tree.pandas.df()

scalevars(df)

suffix = ''
if args.data:
    suffix='_data'
elif args.mc:
    suffix='_MC'
tags = {'pi_fromV0':[0,1], 'p_fromL':[2], 'e_fromconversions':[3], 'kaons_fromkinks':[4], 
'kaons_fromTOF':[5], 'deuterons_fromTOFTPC':[8], 'triton_fromTOFTPC':[9], 'He3_fromTOFTPC':[10]}
for tag, selbits in tags.items():
    df_tag = filter_bit_df(df,selbits)
    df_tag.to_parquet('{0}/{1}{2}.parquet.gzip'.format(args.outdir,tag,suffix),compression='gzip')
