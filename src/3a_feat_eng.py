# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
pd.options.mode.chained_assignment = None 


def read_and_shift():
    """[Returns all features that need to be shifted]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    non_shift = ['player_name', 'position', 'was_home', 'kickoff_time', 'GW', 'team_a', 'team_a_difficulty', 'team',
                'team_h', 'team_h_difficulty', 'strength', 'strength_attack_away', 'strength_attack_home', 
                'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 'strength_overall_home',
               'selected', 'value', 'opponent_team', 'round']
    
    df_train = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//cleaned_fpl.csv')
    df_test = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//training//cleaned_fpl.csv')
    
    for df in df_train, df_test:
        shift = set(df.columns) - set(non_shift) 
        for col in shift:
                df[col + '_shift'] = df[col].shift(1).fillna(0)
        df.drop(shift, axis = 1 , inplace=True)
    return df_train, df_test

def univariate_mse(df, num):
   
    # Todo: Implement cross-validation to find a more reasonable estimate
    
    """
    First, it builds one decision tree per feature, to predict the target
    Second, it makes predictions using the decision tree and the mentioned feature
    Third, it ranks the features according to the machine learning metric (roc-auc or mse)
    It selects the highest ranked features
    """

    df = df.select_dtypes(include='number') 
    X_train, X_test, y_train, y_test= train_test_split(df.drop(columns = ['total_points_shift']), 
                                                       df['total_points_shift'],
                                                       test_size=0.2,
                                                       random_state=0) 
    mse_values = []
    for feature in X_train.columns:
        clf = DecisionTreeRegressor(random_state=0)
        clf.fit(X_train[feature].to_frame(), y_train)
        y_scored = clf.predict(X_test[feature].to_frame())
        mse_values.append(mean_squared_error(y_test, y_scored))
    mse_values = pd.Series(mse_values)
    mse_values.index = X_train.columns
    mse_values.sort_values(ascending=True)    
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
    data[new_feats] = data[feats].rolling(min_periods=1, window=prev_games).mean()
    return data

def change_different_teams(df):
    changed_teams = ['Fulham', 'Leeds', 'West Brom', 'Watford', 'Bournemouth', 'Norwich'] # These teams are not in both seasons
    for feat in ['team', 'opponent_team', 'team_a', 'team_h']:
        df[feat] = np.where(df[feat].isin(changed_teams), 'Other', df[feat])
    return df

def create_team_stats(df):
    team_stats = df[['team','strength_attack_home', 'strength_attack_away',
                     'strength_defence_home', 'strength_defence_away']].drop_duplicates()
    team_stats.rename(columns = {'team':'opponent_team', 
                                'strength_attack_home':'opponent_strength_attack_home',
                                'strength_attack_away':'opponent_strength_attack_away',
                                'strength_defence_home':'opponent_strength_defence_home',
                                'strength_defence_away':'opponent_strength_defence_away'}, inplace = True)
    
    df = pd.merge(df, team_stats, on=['opponent_team'])
    df['player_team_strength'] = np.where(df['was_home'], df['strength_attack_home'], df['strength_attack_away'])
    df['player_team_defence'] = np.where(df['was_home'], df['strength_defence_home'], df['strength_defence_away'])
    df['player_team_overall'] = df['player_team_strength']/2 + df['player_team_defence']/2
    df['opponent_team_strength'] = np.where(df['was_home'], df['opponent_strength_attack_away'], df['opponent_strength_attack_home'])
    df['opponent_team_defence'] = np.where(df['was_home'], df['opponent_strength_defence_away'], df['opponent_strength_defence_home'])
    df['opponent_team_overall'] = df['opponent_team_strength']/2 + df['opponent_team_defence']/2
    df.drop(team_stats.columns.drop(['opponent_team']), axis=1, inplace=True)
    return df

def create_double_week(df):
    idx = df[df.duplicated(['player_name', 'GW'], False)].index
    df['double_week'] = False
    df.loc[idx, 'double_week'] = True
    df.reset_index(drop = True, inplace=True)
    return df

def one_hot_encode(df_test):
    """[One hot encode all features but the players name and the kickoff time]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    cat = df_test.select_dtypes(exclude='number').columns.drop(['player_name', 'kickoff_time']) 
    df_test = pd.get_dummies(df_test, columns=cat, prefix=cat)
    return df_test

def cumulative_mm(df):
    df['match_num'] = 1
    df['matches_shift_cumulative'] = df.groupby(['player_name'])['match_num'].cumsum()
    df['minutes_shift_cumulative'] = df.groupby(['player_name'])['minutes_shift'].cumsum()
    df.reset_index(drop=True, inplace=True)
    df.drop(columns = 'match_num', inplace = True)
    return df


def ratio_to_value(df, top_5):
    for col in top_5:
        df[col + '_to_value'] = df[col] / df['value']  
        df[col + '_per_minute'] = df[col] / df['minutes_shift']  
    return df.replace([np.inf, -np.inf, np.nan], 0) # Since some players play 0 minutes

def prob_of_being_selected(df):
    df = df.drop_duplicates(subset = ['player_name', 'GW'], keep='last')
    df['weekly_selection']  = df.groupby(['GW'])['selected'].transform('sum') 
    df['prob_of_selected'] = df['selected'] / df['weekly_selection']
    df.drop(columns = 'weekly_selection', inplace = True)
    df.reset_index(drop = True, inplace=True)
    return df.fillna(0)

def create_top_scorer(df):
    df['max_points'] = df.groupby(['GW'])['total_points_shift'].transform('max')
    df['top_scorer'] = np.where(df['total_points_shift'] == df['max_points'], True, False)
    df.drop(columns = 'max_points', inplace=True)
    return df

def feat_eng(df, feat_choice):
    df = df.groupby(['player_name']).apply(rolling_avg, prev_games = 3, feats= feat_choice) #* 2. Rolling average
    df = ratio_to_value(df, feat_choice) #* 3. Top five + total_points to ratio to value and per minute played
    df = create_team_stats(df) #* 4. Created opponent strength, defense and overall statistics
    df = create_double_week(df) #* 5. Created a binary double week
    df = change_different_teams(df) #* 6. Changed seasonal teams to other
    df = one_hot_encode(df) #* 7. One hot encoded all categorical features
    df = cumulative_mm(df) #* 8. Cumulative matches and minutes played 
    df = prob_of_being_selected(df) #* 9. The probability of being selected 
    df = create_top_scorer(df) #* 10. Boolean indicating if he was the top_scorer
    return df
# %%
df_train, df_test = read_and_shift() #* 1: Shift the data
feat_choice = univariate_mse(df_train, 5) #* Highest univariate decrease 
df_train = feat_eng(df_train, feat_choice)
df_test = feat_eng(df_test, feat_choice)
# %%
df_train.columns.to_list()
# %%
