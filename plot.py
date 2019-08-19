import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import file
while True:
    try:
        filename=input('Input file name and press enter: ')    
    except (FileNotFoundError, IOError):
        print("Wrong file or not found")
    else:
        break

#dictionary for pdg code
pdg={'pion':211,'electron':11,'kaons':321,'proton':2212}

#set number particle
while True:
    print(pdg)
    x=input('Input name relative to particle and press enter: ')
    if(x in pdg):
        break

#pdfcode relative to particle
code = pdg[x]
#dataframe
df = pd.read_parquet(filename)
#dataframe with the pdgcode correct
dfpure = df[df.PDGcode==code]

#data for plot
dEits = dfpure['dEdxITS']
pt = dfpure['p']
dEtpc = dfpure['dEdxTPC']
ptpc = dfpure['pTPC']
width = [0, 0.3, 0.5, 0.75, 1, 1.5, 3, 5, 10] #define manually width for bins hist
dfpure_length = len(dfpure.p)

#first figure and plot
plt.figure(0)

#subplot1
plt.subplot(2,1,1)
plt.scatter(pt,dEits,s=10,c='blue',marker='^')
plt.title('ITS')
plt.ylabel('dE/dx')
plt.xlabel('p_total')

#subplot2
plt.subplot(2,1,2)
plt.scatter(ptpc,dEtpc,s=10,c='red',marker='D')
plt.title('TPC')
plt.ylabel('dE/dx')
plt.xlabel('p_TPC')

#add space between plots
plt.tight_layout()

#figure hist
plt.figure(1)

#plot hist
n, bin, patches = plt.hist(pt,bins=width,edgecolor='white') #n is an array of the height of bars
plt.title('p_tot hist')
plt.ylabel('conteggi')
plt.xlabel('pt')

#probability
prob = [0]*len(n)
for i in range(len(n)):
    prob[i]=n[i]/dfpure_length
    print('Probability for p between %.2f and %.2f is: %.2f'%(width[i],width[i+1],prob[i]))

#show plots and figures
plt.show()
