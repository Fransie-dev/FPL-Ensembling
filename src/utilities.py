import pandas as pd
import os
import shutil

def missing_zero_values_table(df):
    """[This function checks for missing values]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
    mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
    mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("The dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
        "There are " + str(mz_table.shape[0]) +
            " columns that have missing values.")
    return mz_table
    
    
def dlt_create_dir(path):
    """[This function deletes (if existing) and creates a directory]

    Args:
        path ([type]): [description]
    """    
    shutil.rmtree(path,ignore_errors=True)
    os.makedirs(path, exist_ok = True)
    
    