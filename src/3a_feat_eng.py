# %%
import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def read_and_shift():
    """[Returns all features that need to be shifted]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    non_shift = ['player_name', 'team', 'position', 'value', 'GW',
       'kickoff_time', 'season', 'was_home', 'opponent_team', 'selected', 
       'transfers_in', 'transfers_out', 'transfers_balance', 'strength_attack_away',
       'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away',
       'strength_overall_home']

    df = pd.read_csv(
        'C://Users//jd-vz//Desktop//Code//data//collected_us.csv')

    shift = set(df.columns) - set(non_shift)
    for col in shift:
        df[col + '_shift'] = df[col].shift(1).fillna(0)
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
                                'min_samples_split': range(10, 60, 10)},
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


def create_team_stats(df):
    team_stats = df[['team', 'strength_attack_home', 'strength_attack_away',
                     'strength_defence_home', 'strength_defence_away', 'season']].drop_duplicates()
    team_stats.rename(columns={'team': 'opponent_team', 'strength_attack_home': 'opponent_strength_attack_home',
                               'strength_attack_away': 'opponent_strength_attack_away', 'strength_defence_home': 'opponent_strength_defence_home',
                               'strength_defence_away': 'opponent_strength_defence_away'}, inplace=True)

    df = pd.merge(df, team_stats, on=['opponent_team', 'season'])
    df['player_team_strength'] = np.where(df['was_home'], df['strength_attack_home'], df['strength_attack_away'])
    df['player_team_defence'] = np.where(df['was_home'], df['strength_defence_home'], df['strength_defence_away'])
    df['player_team_overall'] = df['player_team_strength'] / 2 + df['player_team_defence']/2
    df['opponent_team_strength'] = np.where(df['was_home'], df['opponent_strength_attack_away'], df['opponent_strength_attack_home'])
    df['opponent_team_defence'] = np.where(df['was_home'], df['opponent_strength_defence_away'], df['opponent_strength_defence_home'])
    df['opponent_team_overall'] = df['opponent_team_strength'] / 2 + df['opponent_team_defence']/2
    return df


def create_FDR(df):
    df['FDR'] = np.where(df['opponent_team_overall'] >
                         df['player_team_overall'], 'High', 'Low') # High, Low
    idx_med = np.where(df['opponent_team_overall']
                       == df['player_team_overall'])
    df['FDR'].iloc[idx_med] = 'Med' # Mid
    
    drop_feat = ['opponent_strength_attack_home', 'opponent_strength_attack_away', 'opponent_strength_defence_home',
                    'opponent_strength_defence_away', 'player_team_strength','player_team_defence', 'player_team_overall',
                    'opponent_team_strength', 'opponent_team_defence', 'opponent_team_overall', 'strength_attack_away',
                    'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 'strength_overall_home']

    df.drop(columns = drop_feat, axis=1, inplace=True)
    return df

def create_supporters(df):
    df['Supporters'] = np.where(df['kickoff_time'] < '2020-03-13', True, False)
    return df
    
def create_double_week(df):
    idx = df[df.duplicated(['player_name', 'GW'], False)].index
    df['double_week'] = False
    df.loc[idx, 'double_week'] = True
    df.reset_index(drop=True, inplace=True)
    return df


def one_hot_encode(df):
    print(df.select_dtypes(exclude='number').columns)
    ohe_cols = ['team', 'position', 'opponent_team']
    df = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols)
    for col in df.select_dtypes(exclude='number').columns:
        if (len(df[col].unique()) > 2):
            ohe_cols.append(col)
        else:
            df[col] = df[col].replace({df[col].unique()[0]:0, df[col].unique()[1]:1})
    print(ohe_cols)
    # df = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols)
    return df


# def one_hot_encode(df):
#     from sklearn.preprocessing import OneHotEncoder as ohe
#     from sklearn.preprocessing import LabelBinarizer as le
#     # cat = df.select_dtypes(exclude='number').columns.drop(['player_name', 'kickoff_time', 'season', 'clean_sheets_shift', 'own_goals_shift', 'penalties_missed_shift', 'red_cards_shift', 'yellow_cards_shift'])
#     for col in cat:
#             n = len(df[col].unique())
#             if (n > 2):
#                 df = pd.get_dummies(df, columns=list(col), prefix=list(col))
#             else:
#                 le.fit(df[col])
#                 df[col] = ohe.transform(df[col])
#     return df



def cumulative_mm(df):
    df['match_num'] = 1
    df['matches_shift_cumulative'] = df.groupby(
        ['player_name'])['match_num'].cumsum()
    df['minutes_shift_cumulative'] = df.groupby(
        ['player_name'])['minutes_shift'].cumsum()
    df.reset_index(drop=True, inplace=True)
    df.drop(columns='match_num', inplace=True)
    return df


def ratio_to_value(df, top_5):
    for col in top_5 + ['total_points_shift']:
        df[col + '_to_value'] = df[col] / df['value']
        df[col + '_per_minute'] = df[col] / df['minutes_shift']
    # Since some players play 0 minutes
    return df.replace([np.inf, -np.inf, np.nan], 0)


def prob_of_being_selected(df):
    df = df.drop_duplicates(subset=['player_name', 'GW', 'season'], keep='last')
    df['weekly_selection'] = df.groupby(['GW'])['selected'].transform('sum')
    df['prob_of_selected'] = df['selected'] / df['weekly_selection']
    df['prob_of_selected'] = df['prob_of_selected']
    df.drop(columns='weekly_selection', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


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
feat_choice = univariate_mse(df, 5)  # * Highest univariate decrease
df = feat_eng(df, feat_choice)
df.to_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv', index = False)
# %%
import pandas as pd
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
df['Player_Rank'].value_counts()
df['Team_Rank'].value_counts()
# %%
