# %%
import pandas as pd

def shifted_feats(df):
    """[Returns all features that need to be shifted]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    non_shift = ['player_name', 'position', 'was_home', 'kickoff_time', 'GW', 'team_a', 'team_a_difficulty',
                'team_h', 'team_h_difficulty', 'strength', 'strength_attack_away', 'strength_attack_home',
                'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 'strength_overall_home']
    shift = set(df.columns) - set(non_shift)
    return list(shift)

def shift_rows(df, cols):
    """[This function shifts a selected set of rows ]

    Args:
        df ([type]): [description]
        colNames ([type]): [description]

    Returns:
        [type]: [description]
    """    
    for col in cols:
            if col in df.columns:
                df[col + '_shift'] = df[col].shift(1)
    return df.sort_values(by='GW')

def check_shift():
    training_path = f'C://Users//jd-vz//Desktop//Code//data//2019-20//training//'
    df = pd.read_csv(training_path + 'cleaned_fpl.csv')
    df_shift = pd.read_csv(training_path + 'shifted_fpl.csv')
    a, b = df[df.player_name == 'Aaron Cresswell'][['GW', 'total_points', 'bps']], df_shift[df_shift.player_name == 'Aaron Cresswell'][['GW', 'total_points_shift', 'bps_shift']]
    print(pd.merge(a, b, on='GW'))

def main(season):
    training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
    fpl = pd.read_csv(training_path + 'cleaned_fpl.csv')
    understat = pd.read_csv(training_path + 'cleaned_understat.csv')
    fpl_shift = fpl.groupby(['player_name']).apply(shift_rows, cols = shifted_feats(fpl))
    fpl_shift.to_csv(training_path + 'shifted_fpl.csv', index=False)
    understat_shift = understat.groupby(['player_name']).apply(shift_rows, cols = shifted_feats(understat))
    understat_shift.to_csv(training_path + 'shifted_us.csv', index=False)
    if season is '2020-21':
        check_shift(fpl, fpl_shift, ['GW', 'total_points'])
    
if __name__ == "__main__": 
    main(season='2020-21') 
    main(season='2019-20') 
    print('Success')
# %%