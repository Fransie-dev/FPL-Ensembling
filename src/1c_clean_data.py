# %%
import pandas as pd
 
def drop_dups(df):
    df.drop([col for col in df.columns if col.endswith('_y')], axis=1, inplace=True)
    df.columns = df.columns.str.replace('_x', '')
    identifiers = ['element', 'fixture', 'round', 'season', 'id', 'roster_id']
    teams = ['opponent_team', 'name', 'short_name', 'h_team', 'a_team']
    dups = ['h_goals', 'a_goals', 'goals', 'date', 'time']
    for col in identifiers + teams + dups:
            if col in df.columns:
                df.drop(columns = col, inplace = True)
                
def main(season):
    training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
    fpl = pd.read_csv(training_path + 'fpl.csv').dropna()
    understat = pd.read_csv(training_path + 'understat_merged.csv').dropna()
    drop_dups(fpl)
    drop_dups(understat)
    fpl.sort_values(by='GW').to_csv(training_path + 'cleaned_fpl.csv', index = False)
    understat.sort_values(by='GW').to_csv(training_path + 'cleaned_understat.csv', index = False) 

if __name__ == "__main__": 
    main(season='2020-21') # Successful execution
    main(season='2019-20') # Successful execution
    print('Success!')
    