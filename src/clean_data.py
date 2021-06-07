# %%
import pandas as pd
# season='2020-21'
season='2019-20'
training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//Training Data//'
fpl_df = pd.read_csv(training_path + 'fpl.csv')
understat_df = pd.read_csv(training_path + 'understat_merged.csv')

# %%
subset_1 = fpl_df['']