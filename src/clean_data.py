# %%
import pandas as pd
from merge_data import rename_fpl_teams


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
    return fpl, teams, understat


def change_team_strength(teams, fpl):
    """[This function changes the current constant strength layout to fixture based features]

    Args:
        teams ([type]): [description]
        fpl ([type]): [description]

    Returns:
        [type]: [description]
    """    
    teams = rename_fpl_teams(teams, features=['name'])
    team_h_strength_attack = []
    team_h_strength_defense = []
    team_a_strength_attack = []
    team_a_strength_defense = []
    for i in range(len(fpl)):
            team_h_strength_attack.append(teams[teams['name'] == fpl['team_h'][i]]['strength_attack_home'].values)
            team_h_strength_defense.append(teams[teams['name'] == fpl['team_h'][i]]['strength_defence_home'].values)
            team_a_strength_attack.append(teams[teams['name'] == fpl['team_a'][i]]['strength_attack_away'].values)
            team_a_strength_defense.append(teams[teams['name'] == fpl['team_a'][i]]['strength_defence_away'].values)
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
    fpl.drop(['FPL_ID', 'fixture', 'id','name', 'short_name', 'opponent_team', 'round', 'team', 'strength','strength_attack_away', 
              'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 
              'strength_overall_home'], axis = 1, inplace = True) # Duplicate features
    fpl_df = fpl[['GW', 'player_name', 'total_points', 'was_home',
                  'team_h', 'team_h_difficulty', 'team_h_score', 'team_h_strength_attack', 'team_h_strength_defense', 'team_h_strength_overall',
                  'team_a', 'team_a_difficulty', 'team_a_score', 'team_h_strength_attack', 'team_h_strength_defense', 'team_h_strength_overall',
                  'position',  'goals_scored', 'goals_conceded', 'assists', 'clean_sheets','penalties_saved', 'penalties_missed', 
                  'saves','own_goals', 'red_cards', 'yellow_cards', 'minutes', 
                  'selected', 'value', 'bonus', 'bps', 'transfers_balance', 'transfers_in', 'transfers_out', 'influence', 'creativity', 'threat', 'ict_index',
                  'kickoff_time']].copy()
    fpl_df.sort_values(by='GW',inplace=True)
    return fpl_df

def one_hot_encode(fpl):
    """[This function one hot encodes the four categorical features into dummy variables]

    Args:
        fpl ([type]): [description]

    Returns:
        [type]: [description]
    """    
    print('Before one hot encoding, the data is of shape ', fpl.shape)
    fpl = pd.get_dummies(fpl, columns=['was_home', 'position', 'team_h', 'team_a'], 
                      prefix=['home', 'position', 'team_h', 'team_a'])
    print('After one hot encoding, the data is of shape ', fpl.shape)
    print('The data gained 45 features, and removed four of them')
    fpl.drop(columns=['home_False'], axis=1, inplace=True)
    return fpl

def understat_remove_duplicates_and_id(understat):
    """[summary]

    Args:
        understat ([type]): [description]

    Returns:
        [type]: [description]
    """    
    understat.drop(['FPL_ID', 'fixture', 'id_x','name', 'short_name', 'opponent_team', 'round', 'team', 'strength','strength_attack_away', 
              'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 
              'strength_overall_home', 'position_y', 'h_team', 'a_team', 'id_y', 'season', 'roster_id', 'date',
               'goals', 'h_goals', 'a_goals',  'assists_y', 'time'], axis = 1, inplace = True) # Duplicate features
    
    understat_df = understat[['GW', 'player_name', 'total_points', 'was_home',
                  'team_h', 'team_h_difficulty', 'team_h_score', 'team_h_strength_attack', 'team_h_strength_defense', 'team_h_strength_overall',
                  'team_a', 'team_a_difficulty', 'team_a_score', 'team_h_strength_attack', 'team_h_strength_defense', 'team_h_strength_overall',
                  'position_x',  'goals_scored', 'goals_conceded', 'assists_x', 'clean_sheets','penalties_saved', 'penalties_missed', 
                  'saves', 'own_goals', 'red_cards', 'yellow_cards', 'minutes', 'selected', 'value', 'bonus', 'bps', 'transfers_balance',
                  'transfers_in', 'transfers_out', 'influence', 'creativity', 'threat', 'ict_index', 'kickoff_time',
                  'shots',  'key_passes', 'xG', 'xA', 'npg', 'npxG', 'xGChain', 'xGBuildup']].copy()
    understat_df.sort_values(by='GW',inplace=True)
    understat_df.rename(columns = {'position_x':'position',
                                   'assist_x':'assists'}, inplace = True)
    return understat_df

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
    fpl = one_hot_encode(fpl)
    return fpl
    

def preprocess_understat(understat, teams):
    understat = change_team_strength(teams, understat)
    understat = understat_remove_duplicates_and_id(understat)
    understat = one_hot_encode(understat)
    return understat
    
def main(season):
    training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
    data_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//'
    fpl, teams, understat = read_data(data_path, training_path)
    fpl = preprocess_fpl(fpl, teams)
    understat = preprocess_understat(understat, teams)
    fpl.to_csv(training_path + 'cleaned_fpl.csv')
    understat.to_csv(training_path + 'cleaned_understat.csv')

if __name__ == "__main__":
    main(season='2020-21') # Successful execution
    main(season='2019-20') # Successful execution
    print('Success!')

