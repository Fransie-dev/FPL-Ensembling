# %%
import pandas as pd
# season='2020-21'
season='2019-20'
training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
fpl_df = pd.read_csv(training_path + 'fpl.csv')
understat_df = pd.read_csv(training_path + 'understat_merged.csv')
list(fpl_df.columns)








# %%
def fpl_remove_duplicates_and_id(fpl):
    fpl_df.drop(['FPL_ID', 'fixture', 'id','name','short_name'], axis = 1, inplace = True)
    


'FPL_ID',







# TODO: Create opponent strength, opponent difficulty, 