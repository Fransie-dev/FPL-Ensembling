# %%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def shift_data():
    non_shift = ['player_name', 'team', 'position', 'value', 'GW',
       'kickoff_time', 'season', 'was_home', 'opponent_team', 'selected', 
       'transfers_in', 'transfers_out', 'transfers_balance', 'strength_attack_away',
       'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away',
       'strength_overall_home']

    shift = set(df.columns) - set(non_shift)
    for col in shift:
        df[col + '_shift'] = df.groupby('player_name')[col].shift(1).fillna(0)
    df.drop(shift, axis=1, inplace=True)
    return df

def to_cat(df):
    for col in ['season', 'clean_sheets_shift', 'own_goals_shift', 'penalties_missed_shift', 'red_cards_shift', 'yellow_cards_shift']:
        df[col] = df[col].astype('category')
    return df

def univariate_mse(df, num):
    df = df.select_dtypes(include='number')
    X_train, y_train = df.drop(columns=['total_points_shift']), df['total_points_shift']
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

def rolling_avg(data, prev_games, feats):
    """[This function creates a previous game rolling average for the selected features]

    Args:
        data ([type]): [description]
        prev_games ([type]): [description]
        feats ([type]): [description]

    Returns:
        [type]: [description]
    """
    new_feats = []
    for feat in feats:
        new_feats.append(feat + '_last_' + str(prev_games))
    data[new_feats] = data[feats].rolling(min_periods=1, window=prev_games).mean().fillna(0)
    return data


def change_different_teams(df):
    changed_teams = ['Fulham', 'Leeds', 'West Brom', 'Watford',
                     'Bournemouth', 'Norwich']  # These teams are not in both seasons
    for feat in ['team', 'opponent_team']:
        df[feat] = np.where(df[feat].isin(changed_teams), 'Other', df[feat])
    return df

def create_team_stats(df): # TODO: Updt: 0, doe snot include the teams scores
    """[This function creates FDR, replaces home and away scores with team scores and opponent team score, 
    and also drops irrelevant team strength statistics]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    team_stats = df[['team', 'strength_attack_home', 'strength_attack_away',
                     'strength_defence_home', 'strength_defence_away', 'season']].drop_duplicates()
    
    team_stats.rename(columns={'team': 'opponent_team', 'strength_attack_home': 'opponent_strength_attack_home',
                               'strength_attack_away': 'opponent_strength_attack_away', 
                               'strength_defence_home': 'opponent_strength_defence_home',
                               'strength_defence_away': 'opponent_strength_defence_away'}, inplace=True)

    df = pd.merge(df, team_stats, on=['opponent_team', 'season'])
    df['player_team_strength'] = np.where(df['was_home'], df['strength_attack_home'], df['strength_attack_away'])
    df['player_team_defence'] = np.where(df['was_home'], df['strength_defence_home'], df['strength_defence_away'])
    df['player_team_overall'] = df['player_team_strength'] / 2 + df['player_team_defence']/2
    df['opponent_team_strength'] = np.where(df['was_home'], df['opponent_strength_attack_away'], df['opponent_strength_attack_home'])
    df['opponent_team_defence'] = np.where(df['was_home'], df['opponent_strength_defence_away'], df['opponent_strength_defence_home'])
    df['opponent_team_overall'] = df['opponent_team_strength'] / 2 + df['opponent_team_defence']/2
    df['FDR'] = np.where(df['opponent_team_overall'] > df['player_team_overall'] + 10, 'High',
                     np.where(df['opponent_team_overall'] < df['player_team_overall'] - 10, 'Low', 'Med')) 
    df['team_score'] = np.where(df['was_home'], df['team_h_score'], df['team_a_score']) # Replaces h_team, a_team
    df['opponent_team_score'] = np.where(df['was_home'], df['team_a_score'], df['team_h_score']) # Replaces h_team_ateam
    df['win'] = np.where(df['team_score'] > df['opponent_team_score'], 1, 0)
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
    df_temp = df_temp[['team', 'opponent_team', 'kickoff_time', 'team_win_perc', 'opp_team_win_perc']]
    df = pd.merge(df, df_temp, on=['team', 'opponent_team', 'kickoff_time'])
    df['win'] = np.where(df['team_score'] > df['opponent_team_score'], True, False)
    df['loss'] = np.where(df['opponent_team_score'] > df['team_score'], True, False)
    drop_feat = ['opponent_strength_attack_home', 'opponent_strength_attack_away', 'opponent_strength_defence_home',
                'opponent_strength_defence_away', 'player_team_strength','player_team_defence', 'player_team_overall',
                'opponent_team_strength', 'opponent_team_defence', 'opponent_team_overall', 'strength_attack_away',
                'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 'strength_overall_home',
                'team_h_score', 'team_a_score']
    df = df.drop(columns = drop_feat, axis=1)
    return df

def create_position_stats(df):
    df['str_cut'] = df['position_stat'].astype(str).str[0]
    df_temp  = df.loc[df['position'] == 'MID']
    def_names = df_temp.loc[df_temp['str_cut'] == 'D','player_name'].unique()
    attack_names = df_temp.loc[df_temp['str_cut'] == 'A','player_name'].unique()
    df.loc[df['player_name'].isin(def_names) & (df['position'] == 'MID'), 'position'] = 'DMID'
    df.loc[df['player_name'].isin(attack_names) & (df['position'] == 'MID'), 'position'] = 'AMID'
    # Create substitution boolean
    df['substitution'] = np.where(df['position_stat'] == 'Sub', True,False)
    df['position_stat'].value_counts()
    # Create position location
    df['str_cut_1'] = df['position_stat'].astype(str).str[1]
    df['str_cut_2'] = df['position_stat'].astype(str).str[2]
    df['field_location'] = 'General'
    for feat in ['str_cut_1', 'str_cut_2']:
        df.loc[df[feat] == 'C', 'field_location'] = 'Centre'
        df.loc[df[feat] == 'R', 'field_location'] = 'Right'
        df.loc[df[feat] == 'L', 'field_location'] = 'Left'
    df = df.drop(['str_cut', 'str_cut_1', 'str_cut_2', 'position_stat'], axis = 1)
    return df

def create_supporters(df): 
    df['supporters'] = np.where(df['kickoff_time'] < '2020-03-13', True, False)
    return df

def common_selected(df):
    df['selected_qnt'] = df.groupby(['GW','position', 'season'])['selected'].transform('quantile', 0.8)
    df['common_select'] = np.where(df['selected'] > df['selected_qnt'], True, False)
    return df

def transferred(df):
    df['common_transfer'] = np.where(df['transfers_in'] > df['transfers_out'], True, False)
    df.drop(['transfers_in', 'transfers_out', 'transfers_balance'], axis =1, inplace=True)
    return df
    
def create_premium_players(df): 
    df['premium_cutoff'] = df.groupby(['GW', 'season', 'position'])['value'].transform('quantile', 0.75)
    df['medium_cutoff'] = df.groupby(['GW', 'season', 'position'])['value'].transform('quantile', 0.5)
    df['premium_players'] = np.where(df['value'] >= df['premium_cutoff'], 'Premium', np.where(df['value'] >= df['medium_cutoff'], 'Medium', 'Budget'))
    df['over_achiever_cutoff'] =  df.groupby(['GW', 'season', 'position'])['total_points'].transform('quantile', 0.70)
    df['over_achiever'] = np.where(df['premium_players'].isin(['Medium', 'Budget']) & (df['total_points'] > df['over_achiever_cutoff']),  True, False)
    df = df.drop(['over_achiever_cutoff', 'medium_cutoff', 'premium_cutoff'], axis = 1)
    return df

def create_double_week(df):
    df_temp = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_us.csv')[['team', 'GW', 'season', 'kickoff_time']].drop_duplicates()
    df_temp['counter'] = 1 # Num fixtures
    df_temp['num_matches'] = df_temp.groupby(['team', 'GW', 'season'])['counter'].transform('sum') # Num matc hes played per gameweek by team
    df = pd.merge(df, df_temp[['team', 'kickoff_time', 'num_matches']], on=['team', 'kickoff_time'])
    # print(df['num_matches'].value_counts()) # 19712 single, 1334 double, 41 triple
    df['double_week'] = np.where(df['num_matches'] > 1, True, False)
    df.drop('num_matches', axis = 1, inplace=True)
    return df

def cumulative_mm(df):
    df['match_played'] = np.where(df['minutes'] > 60, True, False)
    # df['matches_played'] = df.groupby(['player_name', 'season'])['match_played'].cumsum()
    df['minutes_played'] = df.groupby(['player_name', 'season'])['minutes'].cumsum()
    df.reset_index(drop=True, inplace=True)
    print(df['match_played'].value_counts())
    df.drop(columns=['match_played'], axis = 1, inplace=True)
    return df

def best_teams(df):
    df['gw_team_points'] = df.groupby(['team', 'season', 'GW'])['total_points'].transform('sum')
    df['team_points_cutoff'] = df.groupby(['GW', 'season'])['gw_team_points'].transform('quantile', 0.9)
    df['top_team'] = np.where(df['gw_team_points'] > df['team_points_cutoff'], True, False)
    df['top_player_cutoff'] = df.groupby(['GW', 'season', 'position'])['total_points'].transform('quantile', 0.9)
    df['top_players'] = np.where(df['total_points'] > df['top_player_cutoff'],  True, False)
    df.drop(['gw_team_points', 'team_points_cutoff'], axis=1, inplace=True)
    return df

def ratio_to_value(df):
    df['points_per_mil'] = df['total_points'] / df['value']
    df['points_per_min'] = df['total_points'] / df['minutes']
    return df.replace([np.inf, -np.inf, np.nan], 0)

def create_penalty(df):
    df['penalties_scored'] = df['goals_scored'] - df['npg']
    df['team_penalty'] = np.where(df['penalties_scored'] > 0, True, False)
    df.drop(['penalties_scored', 'npg'], axis = 1, inplace = True)
    return df


    

from feature_selector import *

df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_us.csv')
fs = FeatureSelector(data = df.select_dtypes(include='number').drop('total_points', axis=1), labels = df['total_points'])
fs.identify_collinear(correlation_threshold = 0.8, one_hot = False)
fs.plot_collinear()
df = create_position_stats(df) # * 1. Create team and opponent team scores, update position feature, get field locations
df = create_supporters(df) # * 2. Create supporters
df = create_team_stats(df)
df = common_selected(df) # * 3. Create common transfers
df = transferred(df)
df =  create_premium_players(df) # * 4. Create premium players and overachieving budget players and highest scoring players
df = create_double_week(df) # * Create double week
df = cumulative_mm(df)
df = best_teams(df)
df = ratio_to_value(df)
df = create_penalty(df)
fs = FeatureSelector(data = df.select_dtypes(include='number').drop('total_points', axis=1), labels = df['total_points'])
fs.identify_collinear(correlation_threshold = 0.8, one_hot = False)
fs.plot_collinear()
df.shape
# %%
df.loc[df['penalties_scored'] == -1]
# %%
df[['goals_scored', 'penalties_scored', 'npg', 'team_penalty', 'penalties_missed', 'penalties_saved']].drop_duplicates()
# %%
df.select_dtypes(include='number').corrwith(df['xG']).abs().sort_values()


# %%
def ratio_to_value(df, top_5):
    for col in top_5 + ['total_points_shift']:
        df[col + '_to_value'] = df[col] / df['value']
        df[col + '_per_90'] = df[col] / 90 # 
    # Since some players play 0 minutes
    return df.replace([np.inf, -np.inf, np.nan], 0)


def create_top_scorer(df):
    df['max_points'] = df.groupby(
        ['GW'])['total_points_shift'].transform('max')
    df['top_scorer'] = np.where(
        df['total_points_shift'] == df['max_points'], True, False)
    df.drop(columns='max_points', inplace=True)
    return df

def create_top_team(df):
    df['team_points'] = df.groupby(['team', 'season', 'GW'])[
        'total_points_shift'].transform('sum')
    df['max_points'] = df.groupby(['season', 'GW'])['total_points_shift'].transform('max')
    df['top_team'] = np.where(df['team_points'] == df['max_points'], True, False)
    df.drop(columns=['max_points', 'team_points'], inplace=True)
    return df

def top_3(df, option):
    if option == 'team':
        first, second, third = [],[],[]
        df['team_points'] = df.groupby(['team', 'season', 'GW'])['total_points_shift'].transform('sum') # All points in a week
        for season in [2019,2020]:
            for gw in range(1, 39, 1):
                df_gw = df[(df['GW'] == gw) & (df['season'] == season)]
                for lst in [first, second, third]:
                        df_temp=df_gw[df_gw['team_points']==df_gw['team_points'].max()]
                        lst.append(df_temp.index[0])
                        df_gw = df_gw.drop(df_temp.index[0], axis=0)
        df = df.drop(columns = ['team_points'])
        df['Team_Rank'] = 'No'
        df['Team_Rank'][first] = 'First'
        df['Team_Rank'][second] = 'Second'
        df['Team_Rank'][third] = 'Third'
    if option == 'player':
        first, second, third = [],[],[]
        df['player_points'] = df.groupby(['player_name', 'season', 'GW'])['total_points_shift'].transform('sum') # All points in a week
        for season in [2019,2020]:
            for gw in range(1, 39, 1):
                df_gw = df[(df['GW'] == gw) & (df['season'] == season)]
                for lst in [first, second, third]:
                        df_temp=df_gw[df_gw['player_points']==df_gw['player_points'].max()]
                        lst.append(df_temp.index[0])
                        df_gw = df_gw.drop(df_temp.index[0], axis=0)
        df = df.drop(columns = ['player_points'])
        df['Player_Rank'] = 'No'
        df['Player_Rank'][first] = 'First'
        df['Player_Rank'][second] = 'Second'
        df['Player_Rank'][third] = 'Third'
    return df


def feat_eng(df, feat_choice):
    df['form'] = df['total_points_shift'].rolling(min_periods=1, window=4).mean().fillna(0) # * 1. Calculate 4 week form
    df = df.groupby(['player_name']).apply(rolling_avg, prev_games=4, feats=feat_choice) #* 2. Associated 4 week rolling average of five highest univariate features
    df = ratio_to_value(df, feat_choice)  # * 3. Top five + total_points to ratio to value and per minute played
    df = create_team_stats(df) # * 4. Created opponent strength, defense and overall statistics
    df = create_FDR(df)  # *5. Created a fixture difficulty rating of L, M, H
    df = create_double_week(df)  # * 5. Created a binary double week
    df = change_different_teams(df)  # * 6. Changed seasonal teams to other
    df = top_3(df, 'team') # * Ranks teams in gameweeks
    df = top_3(df, 'player') # * Ranks players in gameweeks
    df = cumulative_mm(df)  # * 8. Cumulative matches and minutes played for a player
    df = create_supporters(df)
    # df = one_hot_encode(df) #* No one hot encoding or scaling just yet
    return df

# %%
df = read_and_shift().pipe(to_cat)  # * 1: Shift the data
feat_choice = ['bps_shift', 'influence_shift', 'bonus_shift', 'goals_scored_shift'] # Obtained from below
df = feat_eng(df, feat_choice)
df.to_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv', index = False)
# %%
import pandas as pd
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
# %%
df = read_and_shift().pipe(to_cat)  # * 1: Shift the data
feat_choice_FWD = univariate_mse(df[df['position'] == 'FWD'], 15)  # * Highest univariate decrease
feat_choice_MID = univariate_mse(df[df['position'] == 'MID'], 15)  # * Highest univariate decrease
feat_choice_DEF = univariate_mse(df[df['position'] == 'DEF'], 15)  # * Highest univariate decrease
feat_choice_GK = univariate_mse(df[df['position'] == 'GK'], 15)  # * Highest univariate decrease
# %%
df_feats = pd.DataFrame([feat_choice_FWD, feat_choice_MID, feat_choice_DEF, feat_choice_GK],
                        index=['FWD', 'MID', 'DEF', 'GK'])
df_feats # bps_shift, bonus_shift, influence_shift, goals_scored_shift
# %%
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_us.csv').pipe(to_cat)  # * 1: Shift the data
# %%
print(df[df['position'] == 'GK'].corrwith(df["total_points_shift"]).abs().sort_values(ascending = False).head(5)) # BPS, Bonus, Clean sheets, influence
print(df[df['position'] == 'FWD'].corrwith(df["total_points_shift"]).abs().sort_values(ascending = False).head(5)) # Clean sheets
print(df[df['position'] == 'DEF'].corrwith(df["total_points_shift"]).abs().sort_values(ascending = False).head(5)) # Bonus
print(df[df['position'] == 'MID'].corrwith(df["total_points_shift"]).abs().sort_values(ascending = False).head(5)) 
# %%
df = read_and_shift().pipe(to_cat).sort_values(['player_name', 'kickoff_time'],ascending = True)
df
# %%
df = read_and_shift()
# %%
