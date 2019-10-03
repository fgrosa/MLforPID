import numpy as np

def single_column(frames):
    '''adding a single column to every dataframe in dict and assigning '''
    #order dataframe by key to avoid errors
    cat_dict  = dict(enumerate(sorted(frames)))
    for keys in frames:
        frames[keys]['category'] = cat_dict[keys]
    print('category assigned :' + str(cat_dict))

def multi_column(frames):
    '''Adding mutiple columns called category to every dataframe in dict and assigning
    boolean value 1 or 0 if it '''
    #matrixidentity of the dimension of length of dictionary
    matrix_category=np.identity(len(frames),dtype=int)
    #converting to numpy array
    array_columns = np.array(sorted(frames),dtype=str)
    #dictionary category
    category_dict = dict ()
    #assigning new column and boolean values
    for category,keys in enumerate(array_columns):
        #create a dict with array columns as keys and assigning a value 0 or 1
        tmp_dict =dict(zip(array_columns, matrix_category[category,:]))
        frames[keys] = frames[keys].assign(**tmp_dict)
        category_dict.update({keys: matrix_category[category,:]})
    print('category assigned: ')
    print(category_dict)



