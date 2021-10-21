# %%
import pandas as pd

def drop_dups(df):
    """[Drop duplicated columns from the merging process]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    init_len = len(df.columns)
    df.drop([col for col in df.columns if col.endswith('_y')], axis=1, inplace=True) 
    df.columns = df.columns.str.replace('_x', '')
    identifiers = ['element', 'fixture', 'id', 'name', 'short_name', 'roster_id']
    teams = ['team_a', 'team_h', 'h_team', 'a_team']
    dups = [ 'round', 'h_goals', 'a_goals', 'goals', 'date', 'time', 
            'season', 'team_a_difficulty',  'team_h_difficulty', 'strength'] 
    for col in identifiers + teams + dups:
            if col in df.columns:
                df.drop(columns = col, inplace = True)
    print(f'Dropped {init_len - len(df.columns)} columns')
    return df

def add_season(df, season):
    """[Assigns a constant season for easier referencing]

    Args:
        df ([type]): [description]
        season ([type]): [description]

    Returns:
        [type]: [description]
    """    
    if season == '2019-20':
        df['season'] = 2019
    if season == '2020-21':
        df['season'] = 2020
    return df


def merge_seasons(sort_by):
    season = '2019-20'
    training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
    df_1 = pd.read_csv(training_path + 'cleaned_fpl.csv')
    df_3= pd.read_csv(training_path + 'cleaned_understat.csv')
    season='2020-21'
    training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
    df_2 = pd.read_csv(training_path + 'cleaned_fpl.csv')
    df_4 = pd.read_csv(training_path + 'cleaned_understat.csv')

    fpl_cols=['player_name','team','position','value','minutes','bps','GW','kickoff_time','season','total_points','creativity','influence','threat','ict_index','assists','bonus','goals_conceded','goals_scored','saves','own_goals','penalties_saved','penalties_missed','red_cards','yellow_cards','team_h_score','team_a_score','clean_sheets','was_home','opponent_team','selected','transfers_in','transfers_out','transfers_balance','strength_attack_away','strength_attack_home','strength_defence_away','strength_defence_home','strength_overall_away','strength_overall_home']
    us_cols = fpl_cols + ['position_stat', 'shots','key_passes','xG','xA','npg','npxG','xGChain','xGBuildup']
    pd.concat([df_1, df_2]).sort_values(by=sort_by).reindex(columns = fpl_cols).to_csv('C://Users//jd-vz//Desktop//Code//data//collected_fpl.csv', index = False)
    pd.concat([df_3, df_4]).sort_values(by=sort_by).reindex(columns = us_cols).to_csv('C://Users//jd-vz//Desktop//Code//data//collected_us.csv', index = False)
    
def main(season):
    training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
    fpl = pd.read_csv(training_path + 'fpl.csv').pipe(drop_dups).pipe(add_season, season)
    fpl.sort_values(by=['GW', 'season']).to_csv(training_path + 'cleaned_fpl.csv', index = False)
    understat = pd.read_csv(training_path + 'understat_merged.csv')
    understat['position_stat'] = understat['position_y']
    understat.pipe(drop_dups).pipe(add_season, season).sort_values(by=['GW', 'season']).to_csv(training_path + 'cleaned_understat.csv', index = False)

if __name__ == "__main__": 
    main(season='2020-21') # Successful execution
    main(season='2019-20') 
    merge_seasons(sort_by = ['GW', 'season', 'player_name']) 
    print('Success!')
