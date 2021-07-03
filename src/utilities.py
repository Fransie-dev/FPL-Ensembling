import pandas as pd
import os
import shutil

def set_season_time(season):
    """[This function specifies the start and ending dates of the season]

    Args:
        season ([type]): [description]

    Returns:
        [type]: [description]
    """    
    if season == '2020-21':
        startdate = time.strptime('12-08-2020', '%d-%m-%Y')
        startdate = datetime.fromtimestamp(mktime(startdate))
        enddate = time.strptime('26-07-2021', '%d-%m-%Y')
        enddate = datetime.fromtimestamp(mktime(enddate))
    if season == '2019-20':
        startdate = time.strptime('09-08-2019', '%d-%m-%Y')
        startdate = datetime.fromtimestamp(mktime(startdate))
        enddate = time.strptime('26-07-2020', '%d-%m-%Y')
        enddate = datetime.fromtimestamp(mktime(enddate))
    return startdate, enddate

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
    
def delete_any_duplicates(df):
    """[Hardcoded solution to a problem within the code]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    df1 = df[df.columns[~df.columns.str.endswith(tuple([str(i) for i in range(10)]))]]
    return df1

def one_hot_encode(fpl):
    """[This function one hot encodes the four categorical features into dummy variables]

    Args:
        fpl ([type]): [description]

    Returns:
        [type]: [description]
    """    
    fpl = pd.get_dummies(fpl, columns=['was_home', 'position', 'team_h', 'team_a'], 
                      prefix=['home', 'position', 'team_h', 'team_a'])
    fpl.drop(columns=['home_False'], axis=1, inplace=True)
    return fpl
