# %%
import pandas as pd
from merge_data import rename_fpl_teams


def read_data(data_path, training_path):
    fpl = pd.read_csv(training_path + 'fpl.csv')
    teams = pd.read_csv(data_path + 'teams.csv')
    teams = rename_fpl_teams(teams, features=['name'])
    understat = pd.read_csv(training_path + 'understat_merged.csv')
    return fpl, teams, understat


def fpl_remove_duplicates_and_id(fpl):
    """[This function removes all the duplicates and identifiers present within the data, and resorts it into a more logical order]

    Args:
        fpl_df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    fpl.drop(['FPL_ID', 'fixture', 'id','name', 'short_name', 'opponent_team', 'round', 'team'], axis = 1, inplace = True) # Duplicates
    fpl_df = fpl[['GW', 'kickoff_time', 'was_home', 'team_h', 'team_h_difficulty', 'team_h_score', 'team_a', 'team_a_difficulty', 'team_a_score',
                     'player_name', 'position', 'total_points', 'goals_scored', 'goals_conceded', 'assists', 'clean_sheets','penalties_saved', 'penalties_missed', 'saves','own_goals', 'red_cards', 'yellow_cards', 'minutes', 
                     'strength','strength_attack_away','strength_attack_home','strength_defence_away','strength_defence_home','strength_overall_away','strength_overall_home',
                     'selected', 'value', 'bonus', 'bps', 'transfers_balance', 'transfers_in', 'transfers_out', 'influence', 'creativity', 'threat', 'ict_index']].copy()
    fpl_df.sort_values(by='GW',inplace=True)
    return fpl_df
# %%
# TODO: Create opponent strength, opponent difficulty, 
# TODO: One hot encode position, team



season='2019-20'
training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
data_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//'
fpl, teams, understat = read_data(data_path, training_path)
fpl_df = fpl_remove_duplicates_and_id(fpl)
