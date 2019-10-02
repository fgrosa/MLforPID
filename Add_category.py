import numpy as np

def single_column(frames):
    '''adding a single column to every dataframe in dict and assigning '''
    #order dataframe by key to avoid errors
    frames = dict(sorted(frames.items()))
    cat_dict  = dict(enumerate(frames.keys()))
    for keys in frames:
        frames[keys]['category'] = cat_dict[keys]
    print('category assigned :' + str(cat_dict))
    
def multi_column(frames):
    '''Adding mutiple columns called category to every dataframe in dict and assigning
    boolean value 1 or 0 if it '''
    #order dataframe by key to avoid errors
    frames = dict(sorted(frames.items()))
    #matrixidentity of the dimension of length of dictionary
    matrix_category=np.identity(len(frames),dtype=int)
    #creation of the new columns for dataframe 
    list_columns = list(frames.keys())
    #converting to numpy array
    array_columns = np.array(list_columns,dtype=str)
    #dictionary category
    category_dict = dict ()
    #assigning new column and boolean values
    for category,keys in enumerate(frames.keys()):
        #create a dict with array columns as keys and assigning a value 0 or 1 
        tmp_dict =dict(zip(array_columns, matrix_category[category,:]))
        frames[keys] = frames[keys].assign(**tmp_dict)
        category_dict.update({keys: matrix_category[category,:]})
    print('category assigned: ')
    print(category_dict)

    
    
