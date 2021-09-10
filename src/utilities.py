import pandas as pd
import os
import shutil
from datetime import datetime
import time
from time import mktime

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


def split_data_by_GW(df, df_test, GW = 1):
    """[This function splits the data according to ]

    Args:
        df ([type]): [description]
        df_test ([type]): [description]
        GW (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """    
    df = df.append(df_test[df_test['GW'] < GW])
    df_test = df_test[df_test['GW'] == GW] 
    X_train = df.drop(columns = ['player_name', 'GW', 'total_points']) #, 'creativity', 'ict_index', 'influence', 'threat'])
    y_train = df['total_points']
    X_test = df_test.drop(columns = ['player_name', 'GW', 'total_points']) #, 'creativity', 'ict_index', 'influence', 'threat'])
    y_test = df_test['total_points']
    return X_train, y_train, X_test, y_test


def impute_by_player(df):
    # Note: Discontinued
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




# def fpl_remove_duplicates_and_id(fpl):
#     """[This function removes all the duplicates and identifiers present within the data, and resorts it into a more logical order]

#     Args:
#         fpl_df ([type]): [description]

#     Returns:
#         [type]: [description]
#     """   
#     print(fpl.columns)

#     fpl_df = fpl[['GW', 'player_name', 'total_points', 'was_home',
#                   'team_h', 'team_h_difficulty', 'team_h_score', 'team_h_strength_attack', 'team_h_strength_defense', 'team_h_strength_overall',
#                   'team_a', 'team_a_difficulty', 'team_a_score', 'team_a_strength_attack', 'team_a_strength_defense', 'team_a_strength_overall',
#                   'position',  'goals_scored', 'goals_conceded', 'assists', 'clean_sheets','penalties_saved', 'penalties_missed', 
#                   'saves','own_goals', 'red_cards', 'yellow_cards', 'minutes', 
#                   'selected', 'value', 'bonus', 'bps', 'transfers_balance', 'transfers_in', 'transfers_out', 'influence', 'creativity', 'threat', 'ict_index',
#                   'kickoff_time']]
#     fpl_df.sort_values(by='GW',inplace=True)
#     return fpl_df


# def understat_remove_duplicates_and_id(understat):
#     """[summary]

#     Args:
#         understat ([type]): [description]

#     Returns:
#         [type]: [description]
#     """    
#     understat_df = understat[['GW', 'player_name', 'total_points', 'was_home',
#                   'team_h', 'team_h_difficulty', 'team_h_score', 'team_h_strength_attack', 'team_h_strength_defense', 'team_h_strength_overall',
#                   'team_a', 'team_a_difficulty', 'team_a_score', 'team_a_strength_attack', 'team_a_strength_defense', 'team_a_strength_overall',
#                   'position_x',  'goals_scored', 'goals_conceded', 'assists_x', 'clean_sheets','penalties_saved', 'penalties_missed', 
#                   'saves', 'own_goals', 'red_cards', 'yellow_cards', 'minutes', 
#                   'selected', 'value', 'bonus', 'bps', 'transfers_balance',
#                   'transfers_in', 'transfers_out', 'influence', 'creativity', 'threat', 'ict_index', 
#                   'kickoff_time',
#                   'shots',  'key_passes', 'xG', 'xA', 'npg', 'npxG', 'xGChain', 'xGBuildup']]
#     understat_df.sort_values(by='GW',inplace=True)
#     understat_df.rename(columns = {'position_x':'position',
#                                    'assists_x':'assists'}, inplace = True)
#     return understat_df



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

# fpl = change_team_strength(teams, fpl).dropna()
# understat = change_team_strength(teams, understat).dropna()

# cols = [non_shifted_feats['features'].to_list() + [col for col in df_test.columns if col.endswith('_shift')]]