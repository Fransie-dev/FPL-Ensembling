# %%
import numpy as np
import pandas as pd
from utilities import delete_any_duplicates

def read_data(data_path, training_path):
    """[This function reads the merged data from merge_data.py]

    Args:
        data_path ([type]): [description]
        training_path ([type]): [description]

    Returns:
        [type]: [description]
    """    
    fpl = pd.read_csv(training_path + 'fpl.csv')
    teams = pd.read_csv(data_path + 'teams.csv')
    understat = pd.read_csv(training_path + 'understat_merged.csv')
    understat_miss = pd.read_csv(training_path + 'understat_miss_merged.csv')
    acr_names =  ['Man City', 'Man Utd', 'West Brom', 'Spurs', 'Sheffield Utd', 'West Ham',
                'Wolves', 'Brighton',   'Chelsea', 'Newcastle', 'Everton',  'Fulham',
                'Arsenal', 'Leeds', 'Liverpool', 'Leicester', 'Southampton', 'Crystal Palace',
                'Aston Villa', 'Burnley', 'Watford', 'Bournemouth', 'Norwich']
    team_names = ['Manchester City', 'Manchester United', 'West Bromwich Albion', 'Tottenham', 
                  'Sheffield United', 'West Ham', 'Wolverhampton Wanderers', 'Brighton', 'Chelsea',
                  'Newcastle', 'Everton', 'Fulham', 'Arsenal', 'Leeds', 'Liverpool', 'Leicester', 
                  'Southampton', 'Crystal Palace', 'Aston Villa', 'Burnley', 'Watford', 'Bournemouth', 'Norwich']
    return fpl, teams, understat, acr_names, team_names, understat_miss

def rename_teams(teams, acr_names, team_names):
    """[This function renames the teams so that change_team_strength can index them easily]

    Args:
        teams ([type]): [description]
    """    
    for i in range(len(acr_names)):
        teams.name.loc[teams.name == acr_names[i]] = team_names[i]
    return teams

def change_team_strength(teams, fpl):
    """[This function creates the strength statistics featuress]

    Args:
        teams ([type]): [description]
        fpl ([type]): [description]

    Returns:
        [type]: [description]
    """    
    team_h_strength_attack = []
    team_h_strength_defense = []
    team_a_strength_attack = []
    team_a_strength_defense = []
    for i in range(len(fpl)):
        team_h = fpl.loc[i, 'team_h']
        team_a = fpl.loc[i, 'team_a']
        team_h_strength_attack.append(teams.loc[teams.name == team_h, 'strength_attack_home'].item())
        team_h_strength_defense.append(teams.loc[teams.name == team_h, 'strength_defence_home'].item())
        team_a_strength_attack.append(teams.loc[teams.name == team_a, 'strength_attack_away'].item())
        team_a_strength_defense.append(teams.loc[teams.name == team_a, 'strength_defence_away'].item())
    fpl['team_h_strength_attack'] = team_h_strength_attack
    fpl['team_h_strength_defense'] = team_h_strength_defense
    fpl['team_a_strength_attack'] = team_a_strength_attack
    fpl['team_a_strength_defense'] = team_a_strength_defense
    fpl['team_h_strength_overall'] = fpl['team_h_strength_attack']/2 + fpl['team_h_strength_defense']/2
    fpl['team_a_strength_overall'] = fpl['team_a_strength_attack']/2 + fpl['team_a_strength_defense']/2
    return fpl
        

def fpl_remove_duplicates_and_id(fpl):
    """[This function removes all the duplicates and identifiers present within the data, and resorts it into a more logical order]

    Args:
        fpl_df ([type]): [description]

    Returns:
        [type]: [description]
    """   
    # fpl.drop(['FPL_ID', 'fixture', 'id','name', 'short_name', 'opponent_team', 'round', 'team', 'strength','strength_attack_away', 
    #           'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 
    #           'strength_overall_home'], axis = 1, inplace = True) # Duplicate features 
    fpl_df = fpl[['GW', 'player_name', 'total_points', 'was_home',
                  'team_h', 'team_h_difficulty', 'team_h_score', 'team_h_strength_attack', 'team_h_strength_defense', 'team_h_strength_overall',
                  'team_a', 'team_a_difficulty', 'team_a_score', 'team_a_strength_attack', 'team_a_strength_defense', 'team_a_strength_overall',
                  'position',  'goals_scored', 'goals_conceded', 'assists', 'clean_sheets','penalties_saved', 'penalties_missed', 
                  'saves','own_goals', 'red_cards', 'yellow_cards', 'minutes', 
                  'selected', 'value', 'bonus', 'bps', 'transfers_balance', 'transfers_in', 'transfers_out', 'influence', 'creativity', 'threat', 'ict_index',
                  'kickoff_time']]
    fpl_df.sort_values(by='GW',inplace=True)
    return fpl_df


def understat_remove_duplicates_and_id(understat):
    """[summary]

    Args:
        understat ([type]): [description]

    Returns:
        [type]: [description]
    """    
    # understat.drop(['FPL_ID', 'fixture', 'id_x','name', 'short_name', 'opponent_team', 'round', 'team', 'strength','strength_attack_away', 
    #           'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 
    #           'strength_overall_home', 'position_y', 'h_team', 'a_team', 'id_y', 'season', 'roster_id', 'date',
    #            'goals', 'h_goals', 'a_goals',  'assists_y', 'time'], axis = 1, inplace = True) # Duplicate features
    
    understat_df = understat[['GW', 'player_name', 'total_points', 'was_home',
                  'team_h', 'team_h_difficulty', 'team_h_score', 'team_h_strength_attack', 'team_h_strength_defense', 'team_h_strength_overall',
                  'team_a', 'team_a_difficulty', 'team_a_score', 'team_a_strength_attack', 'team_a_strength_defense', 'team_a_strength_overall',
                  'position_x',  'goals_scored', 'goals_conceded', 'assists_x', 'clean_sheets','penalties_saved', 'penalties_missed', 
                  'saves', 'own_goals', 'red_cards', 'yellow_cards', 'minutes', 
                  'selected', 'value', 'bonus', 'bps', 'transfers_balance',
                  'transfers_in', 'transfers_out', 'influence', 'creativity', 'threat', 'ict_index', 
                  'kickoff_time',
                  'shots',  'key_passes', 'xG', 'xA', 'npg', 'npxG', 'xGChain', 'xGBuildup']]
    understat_df.sort_values(by='GW',inplace=True)
    understat_df.rename(columns = {'position_x':'position',
                                   'assists_x':'assists'}, inplace = True)
    return understat_df

def impute_by_player(df):
    """[Imputes missing features based on player performance]

    Args:
        understat_miss_merged ([type]): [Dataframe with player-specific missing values]

    Returns:
        [type]: []
    """    
    empty_feats = df.columns[df.isnull().any()].tolist() 
    for feat in empty_feats:
        df[feat] = df.groupby("player_name")[feat].transform(lambda x: x.fillna(x.median()))
    return df


def preprocess_fpl(fpl, teams):
    """[This function performs all the preprocessing performed on the pure FPL data]

    Args:
        fpl ([type]): [description]
        teams ([type]): [description]

    Returns:
        [type]: [description]
    """    
    fpl = change_team_strength(teams, fpl) 
    fpl = fpl_remove_duplicates_and_id(fpl)
    # fpl = one_hot_encode(fpl)
    fpl = delete_any_duplicates(fpl)
    return fpl
    

def preprocess_understat(understat, teams):
    """[This function performs all the preprocessing performed on the merged FPL and Understat data]

    Args:
        understat ([type]): [description]
        teams ([type]): [description]

    Returns:
        [type]: [description]
    """    
    understat = change_team_strength(teams, understat)
    understat = understat_remove_duplicates_and_id(understat)
    # understat = one_hot_encode(understat)
    understat = delete_any_duplicates(understat)
    understat = impute_by_player(understat)
    understat.dropna(inplace = True)
    return understat

def main(season):
    training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
    data_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//'
    fpl, teams, understat, acr_names, team_names, understat_miss = read_data(data_path, training_path)
    teams = rename_teams(teams, acr_names, team_names)
    fpl = preprocess_fpl(fpl, teams)
    understat = preprocess_understat(understat, teams)
    understat_miss = preprocess_understat(understat_miss, teams)
    fpl.to_csv(training_path + 'cleaned_fpl.csv')
    understat.to_csv(training_path + 'cleaned_understat.csv')
    understat_miss.to_csv(training_path + 'cleaned_imp.csv')
# %%

if __name__ == "__main__": 
    # Takes ~ 5 min
    main(season='2020-21') # Successful execution
    main(season='2019-20') # Successful execution
    print('Success!')
# %%