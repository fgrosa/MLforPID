import os
import argparse
import numba
import uproot
import pandas as pd

PARSER = argparse.ArgumentParser(description='Arguments')
PARSER.add_argument('--maxentries', type=int, help='max number of entries')
PARSER.add_argument('--data', action='store_true', help='flag to add "data" suffix')
PARSER.add_argument('--mc', action='store_true', help='flag to add "MC" suffix')
PARSER.add_argument('inputfile', metavar='text', default='AnalysisResults.root', \
    help='input root file name')
PARSER.add_argument('inputdir', metavar='text', default='PWGHF_D2H_SystNsigmaPID_pp_kINT7', \
    help='TDirectorFile name in input root file')
PARSER.add_argument('outdir', metavar='text', default='.', help='output directory')
ARGS = PARSER.parse_args()

def scalevars(dataframe):
    dataframe.pT /= 1000
    dataframe.eta /= 1000
    dataframe.phi /= 1000
    dataframe.p /= 1000
    dataframe.pTPC /= 1000
    dataframe.pTOF /= 1000
    dataframe.pHMPID /= 1000
    dataframe.dEdxITS /= 50
    dataframe.dEdxTPC /= 50
    dataframe.ToF /= 10
    dataframe.TrackLength /= 10
    dataframe.StartTimeRes /= 100
    dataframe.HMPIDsig /= 100
    dataframe.HMPIDocc /= 100

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

def filter_bit_df(dataframe, activatedbit):
    array_tag = dataframe.loc[:, 'tag'].values.astype('int')
    res_on = pd.Series([True]*len(array_tag))

    bitmapon = selectbiton(array_tag, activatedbit)
    res_on = pd.Series(bitmapon)

    dataframe_sel = dataframe[res_on.values]
    return dataframe_sel

if '/home' not in ARGS.outdir or '/Users' not in ARGS.outdir or '$HOME' not in ARGS.outdir:
    ARGS.outdir = os.getcwd()+'/'+ARGS.outdir

if not os.path.isdir(ARGS.outdir):
    print('creating output directory {0}'.format(ARGS.outdir))
    os.mkdir(ARGS.outdir)

TREE = uproot.open(ARGS.inputfile)['{0}/fPIDtree'.format(ARGS.inputdir)]
if ARGS.maxentries:
    DF = TREE.pandas.df(entrystop=ARGS.maxentries)
else:
    DF = TREE.pandas.df()

scalevars(DF)

SUFFIX = ''
if ARGS.data:
    SUFFIX = '_data'
elif ARGS.mc:
    SUFFIX = '_MC'
TAGS = {'pi_fromV0':[0,1], 'p_fromL':[2], 'e_fromconversions':[3], 'kaons_fromkinks':[4], \
    'kaons_fromTOF':[5], 'deuterons_fromTOFTPC':[8], 'triton_fromTOFTPC':[9], 'He3_fromTOFTPC':[10]}
for tag, selbits in TAGS.items():
    df_tag = filter_bit_df(DF,selbits)
    df_tag.to_parquet('{0}/{1}{2}.parquet.gzip'.format(ARGS.outdir, tag, SUFFIX), \
        compression='gzip')
