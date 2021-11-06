# %%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from feature_selector import *

def shift_data(df):
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df['form'] = df['total_points']
    df = df.sort_values(['season', 'player_name', 'GW', 'kickoff_time'])
    non_shift = ['player_name', 'team', 'position', 'value', 'GW',
       'kickoff_time', 'season', 'was_home', 'opponent_team', 'selected', 
       'transfers_in', 'transfers_out', 'transfers_balance', 'substitution', 'position_location',
       'supporters', 'FDR', 'common_select', 'common_transfer', 'premium_players', 'double_week', 'total_points'] # Target feature not shifted, all stats used though
    shift = set(df.columns) - set(non_shift)
    for col in shift:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col + '_shift'] = df.groupby('player_name')[col].shift(1).fillna(False)
        elif pd.api.types.is_categorical_dtype(df[col]):
            df[col + '_shift'] = df.groupby('player_name')[col].shift(1).fillna(df[col].mode()[0])
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col + '_shift'] = df.groupby('player_name')[col].shift(1).fillna(0)
    df.drop(shift, axis=1, inplace=True)
    return df

def univariate_mse(df, num):
    df = df.select_dtypes(include='number')
    X_train, y_train = df.drop(columns=['total_points']), df['total_points']
    mse_values = []
    for feature in X_train.columns:
        regressor = DecisionTreeRegressor()
        gs = GridSearchCV(regressor,
                  param_grid = {'max_depth': range(1, 11),
                                'min_samples_split': range(10, 30, 10)},
                  cv=5, scoring='neg_mean_squared_error')

        gs.fit(X_train[feature].to_frame(), y_train)
        print(f'{feature} best parameters', gs.best_params_)
        print(f'{feature} best score', -gs.best_score_)
        mse_values.append(-gs.best_score_)
    mse_values = pd.Series(mse_values)
    mse_values.index = X_train.columns
    return mse_values.sort_values(ascending=True).head(num).index.to_list()

def rolling_avg(df):
    """[This function creates a previous game rolling average for the selected features]

    Args:
        data ([type]): [description]
        prev_games ([type]): [description]
        feats ([type]): [description]

    Returns:
        [type]: [description]
    """
    df['form'] = df['total_points'].rolling(min_periods=1, window=4).mean().fillna(0) # 4 week form for all players
    df['rolling_influence'] = np.where(df['position'].isin(['FWD', 'AMID', 'DMID']), df['influence'].rolling(min_periods=1, window=4).sum().fillna(0), 0)
    df['rolling_goals'] = np.where(df['position'].isin(['FWD', 'AMID', 'DMID']), df['goals_scored'].rolling(min_periods=1, window=4).sum().fillna(0), 0)
    df['rolling_sheets'] = np.where(df['position'].isin(['DEF', 'GK']), df['clean_sheets'].rolling(min_periods=1, window=4).sum().fillna(0), 0)
    df['rolling_conceded'] = np.where(df['position'].isin(['DEF', 'GK']), df['goals_conceded'].rolling(min_periods=1, window=4).sum().fillna(0), 0)
    return df

def change_different_teams(df):
    changed_teams = ['Fulham', 'Leeds', 'West Brom', 'Watford',
                     'Bournemouth', 'Norwich']  # These teams are not in both seasons
    for feat in ['team', 'opponent_team']:
        df[feat] = np.where(df[feat].isin(changed_teams), 'Other', df[feat])
    return df

def create_team_stats(df): # updt: function was merged and is messy... fix me
    """[This function creates FDR, replaces home and away scores with team scores and opponent team score, 
    and also drops irrelevant team strength statistics]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    team_stats = df[['team', 'strength_attack_home', 'strength_attack_away',
                     'strength_defence_home', 'strength_defence_away', 'season', 'kickoff_time']].drop_duplicates()
    
    team_stats.rename(columns={'team': 'opponent_team', 'strength_attack_home': 'opponent_strength_attack_home',
                               'strength_attack_away': 'opponent_strength_attack_away', 
                               'strength_defence_home': 'opponent_strength_defence_home',
                               'strength_defence_away': 'opponent_strength_defence_away',
                               'season':'season'}, inplace=True)
    team_stats.reset_index(inplace=True, drop = True)
    df = pd.merge(df, team_stats, on=['opponent_team', 'season', 'kickoff_time'], how = 'left') # Need to find alternative
    # df = df.drop_duplicates(['player_name', 'opponent_team', 'kickoff_time', 'season', 'total_points'])
    df = df.dropna(subset = ['total_points'])
    df['player_team_strength'] = np.where(df['was_home'], df['strength_attack_home'], df['strength_attack_away'])
    df['player_team_defence'] = np.where(df['was_home'], df['strength_defence_home'], df['strength_defence_away'])
    df['player_team_overall'] = df['player_team_strength'] / 2 + df['player_team_defence']/2
    df['opponent_team_strength'] = np.where(df['was_home'], df['opponent_strength_attack_away'], df['opponent_strength_attack_home'])
    df['opponent_team_defence'] = np.where(df['was_home'], df['opponent_strength_defence_away'], df['opponent_strength_defence_home'])
    df['opponent_team_overall'] = df['opponent_team_strength'] / 2 + df['opponent_team_defence']/2
    df['FDR'] = pd.cut((df['opponent_team_overall'] - df['player_team_overall']), bins = 3, labels = ['Low', 'Medium', 'Hard']) # *1. Create FDR
    df['team_score'] = np.where(df['was_home'], df['team_h_score'], df['team_a_score']) # *2. Replace home_team and away team score with team and opponent team scores
    df['opponent_team_score'] = np.where(df['was_home'], df['team_a_score'], df['team_h_score']) # *2. Replace home_team and away team score with team and opponent team scores
    df['win'] = np.where(df['team_score'] > df['opponent_team_score'], 1, 0) # *3. Create team win and team loss features
    df['loss'] = np.where(df['opponent_team_score'] > df['team_score'], 1, 0)
    df_temp = df.sort_values(['team', 'kickoff_time'])
    df_temp = df_temp[['team', 'opponent_team', 'kickoff_time', 'win', 'loss', 'season']].drop_duplicates()
    df_temp['counter'] = 1
    df_temp['total_team_games'] = df_temp.groupby(['team', 'season'])['counter'].cumsum() # Running team games
    df_temp['total_opponent_games'] = df_temp.groupby(['opponent_team', 'season'])['counter'].cumsum() # Running opponent games
    df_temp['total_team_wins'] = df_temp.groupby(['team', 'season'])['win'].cumsum() # Running team wins
    df_temp['total_opponent_team_wins'] =  df_temp.groupby(['opponent_team', 'season'])['loss'].cumsum() # Running opponent wins
    df_temp['team_win_perc'] = df_temp['total_team_wins']/df_temp['total_team_games']
    df_temp['opp_team_win_perc'] = df_temp['total_opponent_team_wins'] / df_temp['total_opponent_games']
    df_temp = df_temp[['team', 'opponent_team', 'kickoff_time', 'team_win_perc', 'opp_team_win_perc', 'season']].drop_duplicates() # *4. Create team and opponent team winning percentage
    df = pd.merge(df, df_temp, on=['team', 'opponent_team', 'kickoff_time', 'season'], how = 'left')
    df = df.dropna(subset = ['total_points'])
    df['win'] = np.where(df['team_score'] > df['opponent_team_score'], 1, 0)
    df['loss'] = np.where(df['opponent_team_score'] > df['team_score'], 1, 0)
    drop_feat = ['opponent_strength_attack_home', 'opponent_strength_attack_away', 'opponent_strength_defence_home',
                'opponent_strength_defence_away', 'player_team_strength','player_team_defence', 'player_team_overall',
                'opponent_team_strength', 'opponent_team_defence', 'opponent_team_overall', 'strength_attack_away',
                'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 'strength_overall_home',
                'team_h_score', 'team_a_score']
    df = df.drop(columns = drop_feat, axis=1)
    return df

def position_stats(df):
    df['str_cut'] = df['position_stat'].astype(str).str[0]
    df_temp  = df.loc[df['position'] == 'MID']
    def_names = df_temp.loc[df_temp['str_cut'] == 'D','player_name'].unique()
    attack_names = df_temp.loc[df_temp['str_cut'] == 'A','player_name'].unique()
    centre_names = df_temp.loc[df_temp['str_cut'] == 'M','player_name'].unique()
    sub_names = df_temp.loc[df_temp['str_cut'] == 'S','player_name'].unique()
    df['position'] = np.where(df['position'] == 'MID', 'DMID', df['position'])
    df['position'] = np.where((df['position'] == 'DMID') & (df['player_name'].isin(attack_names)), 'AMID', df['position']) # AMID else DMID
    # Create substitution boolean
    df['substitution'] = np.where(df['position_stat'] == 'Sub', True,False)
    df['position_stat'].value_counts()
    # Create position location
    df['str_cut_1'] = df['position_stat'].astype(str).str[1]
    df['str_cut_2'] = df['position_stat'].astype(str).str[2]
    df['position_location'] = 'General'
    for feat in ['str_cut_1', 'str_cut_2']:
        df.loc[(df[feat] == 'C') & (df['position'].isin(['FWD', 'DMID', 'AMID','DEF'])), 'position_location'] = 'Centre'
        df.loc[(df[feat] == 'R') & (df['position'].isin(['FWD', 'DMID', 'AMID','DEF'])), 'position_location'] = 'Right'
        df.loc[(df[feat] == 'L') & (df['position'].isin(['FWD', 'DMID', 'AMID', 'DEF'])), 'position_location'] = 'Left'
    df = df.drop(['str_cut', 'str_cut_1', 'str_cut_2', 'position_stat'], axis = 1)
    return df

def create_supporters(df): 
    df['supporters'] = np.where(df['kickoff_time'] < '2020-03-13', True, False)
    return df

def common_selected(df):
    df['selected_qnt'] = df.groupby(['GW','position', 'season'])['selected'].transform('quantile', 0.8)
    df['common_select'] = np.where(df['selected'] > df['selected_qnt'], True, False)
    df.drop('selected_qnt', axis = 1, inplace=True)
    return df

def transferred(df):
    df['common_transfer'] = np.where(df['transfers_in'] > df['transfers_out'], 1, 0)
    df.drop(['transfers_in', 'transfers_out', 'transfers_balance'], axis =1, inplace=True)
    return df
    
def create_premium_players(df): 
    df['premium_cutoff'] = df.groupby(['GW', 'season', 'position'])['value'].transform('quantile', 0.75)
    df['medium_cutoff'] = df.groupby(['GW', 'season', 'position'])['value'].transform('quantile', 0.5)
    df['premium_players'] = np.where(df['value'] >= df['premium_cutoff'], 'Premium', np.where(df['value'] >= df['medium_cutoff'], 'Medium', 'Budget'))
    premium_qnt = df[df['premium_players'] == 'Premium'].groupby(['GW', 'season', 'position'])['total_points'].quantile(0.75)
    df['over_achiever_cutoff'] = pd.merge(df, premium_qnt, on=['GW', 'season', 'position'], how = 'left')['total_points_y']
    df = df.dropna(subset = ['total_points'])
    df['over_achiever'] = np.where(df['premium_players'].isin(['Medium', 'Budget']) & (df['total_points'] > df['over_achiever_cutoff']),  1, 0)
    df = df.drop(['medium_cutoff', 'premium_cutoff', 'over_achiever_cutoff'], axis = 1)
    return df

def create_double_week(df):
    df_temp = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_us_updated.csv')[['team', 'GW', 'season', 'kickoff_time']].drop_duplicates()
    df_temp['counter'] = 1 # Num fixtures
    df_temp['num_matches'] = df_temp.groupby(['team', 'GW', 'season'])['counter'].transform('sum') # Num matc hes played per gameweek by team
    df = pd.merge(df, df_temp[['team', 'kickoff_time', 'num_matches']], on=['team', 'kickoff_time'], how = 'left') # Trouble child
    df = df.dropna(subset = ['total_points'])
    # df = df.dropna(how = 'any' )
    # print(df['num_matches'].value_counts()) # 19712 single, 1334 double, 41 triple
    df['double_week'] = np.where(df['num_matches'] > 1, True, False)
    df.drop('num_matches', axis = 1, inplace=True)
    return df

def cumulative_mm(df):
    df['played_67'] = np.where(df['minutes'] > 67.5, 1, 0)
    df['minutes_played'] = df.groupby(['player_name', 'season'])['minutes'].cumsum()
    df.reset_index(drop=True, inplace=True)
    # print(df['match_played'].value_counts())
    return df

def best_teams(df):
    df['gw_team_points'] = df.groupby(['team', 'season', 'GW'])['total_points'].transform('sum')
    df['team_points_cutoff'] = df.groupby(['GW', 'season'])['gw_team_points'].transform('quantile', 0.9)
    df['top_team'] = np.where(df['gw_team_points'] > df['team_points_cutoff'], True, False)
    df['top_player_cutoff'] = df.groupby(['GW', 'season', 'position'])['total_points'].transform('quantile', 0.9)
    df['top_players'] = np.where(df['total_points'] > df['top_player_cutoff'],  1, 0)
    df.drop(['gw_team_points', 'team_points_cutoff', 'top_player_cutoff', 'top_team'], axis=1, inplace=True)
    return df

def ratio_to_value(df):
    df['points_per_mil'] = df['total_points'] / df['value']
    df['points_per_min'] = df['total_points'] / df['minutes']
    return df.replace([np.inf, -np.inf, np.nan], 0)

def create_penalty(df):
    df['penalties_scored'] = df['goals_scored'] - df['npg']
    df['team_penalty'] = np.where(df['penalties_scored'] > 0, 1, 0)
    df.drop(['penalties_scored', 'npg'], axis = 1, inplace = True)
    return df

def rolling_avgs(df):
    df = df.groupby(['player_name']).apply(rolling_avg) 
    return df

def sanity_check():
    df_col = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_us.csv')
    df  = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
    df = df[df['season'] ==2020]
    df = df[df['player_name'] == 'Kevin De Bruyne']
    df = df.drop(['total_points'], axis = 1)
    df_col = df_col[df_col['season'] ==2020]
    df_col = df_col[df_col['player_name'] == 'Kevin De Bruyne']
    df_col = df_col[['player_name', 'kickoff_time', 'creativity', 'influence', 'goals_scored', 'total_points']]
    df_hold = pd.merge(df, df_col, on = ['player_name', 'kickoff_time'])
    df_hold = df_hold[['player_name', 'kickoff_time', 'creativity', 'creativity_shift', 'influence', 'influence_shift', 'goals_scored', 'goals_scored_shift',  'total_points']]
    return df_hold
    
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif.sort_values('VIF', ascending = False).reset_index(drop = True)
    return(vif)

def avg_per_match(df, prev = 4, params = []):
    match_params = [para + '_last_' + str(prev) for para in params]
    df['minutes' + '_last_' + str(prev)] = df.groupby('player_name')['minutes'].rolling(min_periods=1, window=prev).sum().fillna(0).reset_index().set_index('level_1').drop('player_name',axis=1)
    df['matches' + '_last_' + str(prev)] = df['minutes' + '_last_' + str(prev)] / 90
    df[match_params] = df.groupby('player_name')[params].rolling(min_periods=1, window=prev).sum().fillna(0).reset_index().set_index('level_1').drop('player_name',axis=1)
    df[match_params].divide(df['matches' + '_last_' + str(prev)], axis=0)
    return df

def find_best_rollback(df, feats):
    for feat in feats:
        window_size = list(range(2,38))
        new_feat = [feat + '_rolling_' + str(i) for i in window_size]
        total_feats = [feat]
        for wind, rollback in enumerate(new_feat):
            df[rollback + '_sum'] = df.groupby(['player_name', 'season'])[feat].rolling(min_periods=1, window=(wind + min(window_size))).sum().droplevel(level=[0, 1])
            total_feats.append(rollback + '_sum')
            df[rollback + '_mean'] = df.groupby(['player_name', 'season'])[feat].rolling(min_periods=1, window=(wind + min(window_size))).mean().droplevel(level=[0,1])
            total_feats.append(rollback + '_mean')
        keep = df[df['season'] < 2021][total_feats].corrwith(df[df['season'] < 2021]['total_points']).idxmax()
        improved_correlation = df[total_feats].corrwith(df['total_points']).max()
        # df[keep] = np.where(df[keep].min() < df[feat].min()/4, 0.25*df[feat].min(), df[keep])
        # df[keep] = np.where(df[keep].max() > 4*df[feat].max(), 4*df[feat].max(), df[keep])
        # print([feature for feature in total_feats if feature not in keep]) # No response
        total_feats.remove(keep) # + Response
        df.drop(total_feats, axis = 1, inplace=True) 
        print(f'{keep} = {improved_correlation}')
    return df

def main():
    # *1. Create FDR
    # *2. Replace home_team and away team score with team and opponent team scores
    # *3. Create team win and team loss features
    # *4. Create team and opponent team winning percentage
    df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_us_updated.csv')
    df.sort_values(by = ['season', 'GW', 'player_name', 'kickoff_time'], inplace = True)
    df = create_team_stats(df)
    #* 5. AMID + DMID
    #* 6. Location
    #* 7. Substitutions
    df = position_stats(df) 
    #* 8. Supporters
    df = create_supporters(df)
    # * 9. Replace transfers in, transfers out, transfers balance with boolean
    df = transferred(df)
    # * 10. Create premium players 
    # * 11. Create overachieving budget and medium-priced players
    df =  create_premium_players(df) 
    # * 12. Create double week
    df = create_double_week(df)
    # * 13. Longer than 67.5 min played
    # * 14. Cumulative minutes played
    df = cumulative_mm(df)
    # * 15. Boolean if a player scored in the top 10th percentile of points in the previous gameweek
    df = best_teams(df)
    # * 16. Replaces npg with penalty boolean
    df = create_penalty(df)
    # * 22. Create last week's form and shift all features unavailable 
    df = shift_data(df) 
    # * 23. Change the different teams to be consistent
    df = change_different_teams(df) 
    # df['team'] =df['team'].astype('str') + '_' + df['season'].astype('str')
    # df['opponent_team'] = df['opponent_team'].astype('str') + '_' + df['season'].astype('str')
    # df['player_name'] = df['player_name'].astype('str') + '_' + df['season'].astype('str')
    # * Write to CSV
    df.sort_values(by = ['season', 'GW', 'player_name',  'kickoff_time']).to_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv', index = False)
    print(df.select_dtypes(include = 'number').corrwith(df['total_points']).abs().sort_values(ascending = False).head(10).to_latex())
    # * 24 Find best rollback features with indicators
    df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
    feats = df.select_dtypes(include='number').drop(['GW', 'season'], axis = 1).columns
    df_roll = find_best_rollback(df, feats)
    df_roll.round(7).sort_values(by = ['season', 'GW', 'player_name', 'kickoff_time']).to_csv('C://Users//jd-vz//Desktop//Code//data//rollbacked_us_testing.csv', index = False)
    print(df_roll.select_dtypes(include = 'number').corrwith(df['total_points']).abs().sort_values(ascending = False).head(10).to_latex())

if __name__ == '__main__':
    df = main()
    sanity_check()
    # calculate_vif(df.select_dtypes(include='number').drop(['ict_index_shift', 'GW', 'total_points', 'season'], axis = 1))
# %%
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//rollbacked_us_testing_4.csv')
df.describe()
# %%
df.select_dtypes(include = 'number').columns
# %%
a = df.groupby('season')['GW'].value_counts().sort_index()
a# %%
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
df['xA_shift']
# %%
print(a)
# %%
