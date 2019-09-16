import pandas as pd
import numpy as np

def add_category(frames):
    cat_array = np.array(range(len(frames.keys())))
    cat_dict  = dict(zip(frames.keys(),cat_array))
    print(cat_dict)
    for keys in frames:
        frames[keys]['category'] = cat_dict[keys]
    
    
