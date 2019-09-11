import pandas as pd
import numpy as np

def add_category(frame, n_category):
    categories = np.zeros(len(frame),dtype=int)
    values  = np.array(range(n_category),dtype=int)
    #replace values in category from 1
    for ind,value in enumerate(values,1):
        start = int(len(frame)*(ind-1)/n_category)
        end = int((len(frame)*ind/n_category))
        categories[start:end] = value
    #add column category
    frame['category'] = categories
    
