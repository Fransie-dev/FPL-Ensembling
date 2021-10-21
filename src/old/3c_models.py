# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
import sklearn.metrics as metrics
from keras.backend import clear_session
from keras.layers import Dense
from keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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


def plot_results(true_value, predicted_value, title):
    plt.figure(figsize=(10,10))
    plt.scatter(true_value, predicted_value, c='crimson')
    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.title(title)
    plt.axis('equal')
    plt.show()

    
def regression_results(y_true, y_pred, model_string):
    print('# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>')
    print(model_string)
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

def scaled_metrics(regressor, X_train, y_train, X_test, y_test, scl_y, GW, model_str, epochs = None):
    if model_str is 'Scaled Net':
        regressor.fit(X_train, y_train, epochs=50)
    else:
        regressor.fit(X_train, y_train)
        if model_str == 'Scaled RF':
            plot_feature_importance(regressor.feature_importances_, X_train.columns,'RANDOM FOREST')
    y_train_rescaled = np.round(scl_y.inverse_transform(y_train))
    y_test_rescaled = np.round(scl_y.inverse_transform(y_test))
    y_pred_test = np.round(scl_y.inverse_transform(regressor.predict(X_test)))
    y_pred_train = np.round(scl_y.inverse_transform(regressor.predict(X_train)))
    regression_results(y_test_rescaled, y_pred_test, model_str)
    # plot_results(y_test_rescaled, y_pred_test, title = f'Testing, 2020 Gameweek {GW}')
    regression_results(y_train_rescaled, y_pred_train, model_str)
    # plot_results(y_train_rescaled, y_pred_train, title = f'Training, 2019 + 2020 GW: {GW - 1}')

def test_scaled_LR_model(X_train, y_train, X_test, y_test, scl_y, GW):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    
    regressor = LinearRegression(normalize=False, n_jobs=-1)
    scaled_metrics(regressor, X_train, y_train, X_test, y_test, scl_y, GW, 'Scaled LR')
    

def test_scaled_knn_model(X_train, y_train, X_test, y_test, scl_y, GW):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    for k in [1, 3, 5, 7]:
        print('k = ', k, end = '\n')
        regressor = KNeighborsRegressor(n_neighbors=k, weights = 'distance')
        scaled_metrics(regressor, X_train, y_train, X_test, y_test, scl_y, GW, 'Scaled kNN')
        
def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'].head(10), y=fi_df['feature_names'].head(15))
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()
            
def test_scaled_RF_model(X_train, y_train, X_test, y_test, std_scale_Y, GW):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    regressor = RandomForestRegressor()  
    scaled_metrics(regressor, X_train, y_train, X_test, y_test, std_scale_Y, GW, 'Scaled RF')


    
def test_scaled_SVR_model(X_train, y_train, X_test, y_test, std_scaler_Y, GW):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    regressor = SVR(verbose=True, kernel='linear') # Note: This was fitted to scaled data
    scaled_metrics(regressor, X_train, y_train, X_test, y_test, std_scaler_Y, GW, 'Scaled SVR')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')
    plt.legend()
    plt.ylim([0,0.05])
    
    
def test_scaled_net(X_train, y_train, X_test, y_test, std_scale_Y, GW):
    """[This is very much just a]

    Args:
        X_train ([type]): [description]
        y_train ([type]): [description]
        X_test ([type]): [description]
        y_test ([type]): [description]
        std_scale_Y_train ([type]): [description]
        std_scale_Y_test ([type]): [description]
        GW ([type]): [description]
    """    
    regressor = Sequential()
    regressor.add(Dense(1024, activation= "relu",))
    regressor.add(Dense(1024, activation= "relu"))
    regressor.add(Dense(1))
    regressor.compile(loss= "mse" , optimizer="adam", metrics=["mse"])
    history = regressor.fit(X_train, y_train, epochs=100, validation_split=0.2)
    y_train_rescaled = np.round(std_scale_Y.inverse_transform(y_train))
    y_test_rescaled = np.round(std_scale_Y.inverse_transform(y_test))
    y_pred_test = np.round(std_scale_Y.inverse_transform(regressor.predict(X_test)))
    y_pred_train = np.round(std_scale_Y.inverse_transform(regressor.predict(X_train)))
    regression_results(y_test_rescaled, y_pred_test, 'Neural Net')
    plot_results(y_test_rescaled, y_pred_test, title = f'Testing, 2020 Gameweek {GW}')
    regression_results(y_train_rescaled, y_pred_train, 'Neural Net')
    plot_results(y_train_rescaled, y_pred_train, title = f'Training, 2019 + 2020 GW: {GW - 1}')
    plot_history(history)
    clear_session()
 
def scale_datasets(df):
  scaled_cols = []
  for col in df.columns.drop('total_points'):
    if df[col].nunique() > 2:
      scaled_cols.append(col)
  std_pred, std_resp = StandardScaler(), StandardScaler()
  X_train, y_train = df.drop('total_points', axis=1), df['total_points'] 
  X_train[scaled_cols] = pd.DataFrame(std_pred.fit_transform(X_train[scaled_cols]), 
                                      columns=df[scaled_cols].columns,
                                      index=X_train.index)
  y_train = pd.DataFrame(std_resp.fit_transform(y_train.to_numpy().reshape(-1, 1)),
                                                columns=['total_points'], index=y_train.index)
  return X_train, y_train, std_resp

def dont_scale_datasets(df):
  X_train, y_train = df.drop('total_points', axis=1), df['total_points'] 
  return X_train, y_train

def preprocess_data(df):
  # Drop indexing features
  idxrs = df[['player_name', 'kickoff_time', 'season', 'GW']]
  df = df.drop(columns = ['player_name', 'kickoff_time', 'season', 'GW'])
  # Binary encoding
  for col in df.columns:
    df[col] = df[col].replace({True:1, False:0})
  # One hot encodings
  ohe_cols = []
  for col in df.select_dtypes(include='object').columns:
      ohe_cols.append(col)
  df = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols)
  df['kickoff_time'] = idxrs['kickoff_time']
  df['GW'] = idxrs['GW']
  return df


def get_training_data(GW, scale):
    df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
    df.drop(['team', 'opponent_team'], inplace = True, axis = 1)
    df = preprocess_data(df)
    min_date = df[(df['GW'] == GW) & (df['kickoff_time'] >= '2020-08-12')]['kickoff_time'].min() # The first date of the gameweek
    max_date = df[(df['GW'] == GW) & (df['kickoff_time'] > '2020-08-12')]['kickoff_time'].max() # The last date of the gameweek
    df_scl = df[df['kickoff_time'] <= max_date].copy() # Only includes up to gameweek 
    std_x, std_y = StandardScaler(), StandardScaler()
    scaled_cols = []
    if scale == True:
      for col in df.columns.drop(['total_points', 'kickoff_time']):
        if df[col].nunique() > 2:
          scaled_cols.append(col)
      df_scl[scaled_cols] = std_x.fit_transform(df_scl[scaled_cols])
      df_scl['total_points'] = std_y.fit_transform(df_scl['total_points'].to_numpy().reshape(-1, 1))
      df_train, df_test = df_scl[df_scl['kickoff_time'] < min_date], df_scl[df_scl['kickoff_time'] >= min_date] 
      X = df_train.drop(['total_points', 'kickoff_time', 'GW'], axis = 1)
      y = df_train['total_points']
      X_val = df_test.drop(['total_points', 'kickoff_time', 'GW'], axis = 1)
      y_val = df_test['total_points']
    else:
      df_train, df_test = df_scl[df_scl['kickoff_time'] < min_date], df_scl[df_scl['kickoff_time'] >= min_date] 
      X = df_train.drop(['total_points', 'kickoff_time', 'GW'], axis = 1)
      y = df_train['total_points']
      X_val = df_test.drop(['total_points', 'kickoff_time', 'GW'], axis = 1)
      y_val = df_test['total_points']
    return X, X_val, y, y_val, std_x, std_y

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Features
univariate_mse_remove = ['selected', 'transfers_in_shift', 'transfers_out_shift', 'transfers_balance_shift']
quasi_remove = ['penalties_saved_shift', 'penalties_missed_shift', 'own_goals_shift', 'red_cards_shift'] 
boruta_select= ['round' ,'value' ,'clean_sheets_shift','goals_scored_shift','saves_shift','assists_shift','bps_shift','influence_shift' ,'yellow_cards_shift' ,'creativity_shift','own_goals_shift'  ,       'red_cards_shift' ,        'minutes_shift' ,          'bonus_shift'   ,          'ict_index_shift' ,        'threat_shift' ,           'goals_conceded_shift',    'position_DEF' ,           'position_FWD',            'position_GK'  ,           'position_MID']        
fwd_select = ['threat_shift', 'bonus_shift', 'creativity_shift', 'influence_shift', 'minutes_shift', 'goals_conceded_shift', 'clean_sheets_shift', 'assists_shift', 'goals_scored_shift', 'yellow_cards_shift', 'bps_shift', 'penalties_saved_shift', 'saves_shift', 'red_cards_shift', 'own_goals_shift', 'position_MID', 'ict_index_shift', 'penalties_missed_shift', 'position_DEF', 'strength_defence_home', 'value', 'team_Man City', 'team_Newcastle', 'team_Arsenal', 'team_Chelsea', 'team_h_Sheffield Utd']
bwd_select = ['value', 'strength_defence_home', 'clean_sheets_shift', 'goals_scored_shift', 'penalties_saved_shift', 'saves_shift', 'assists_shift', 'bps_shift', 'influence_shift', 'penalties_missed_shift', 'yellow_cards_shift', 'creativity_shift', 'own_goals_shift', 'red_cards_shift', 'minutes_shift', 'bonus_shift', 'ict_index_shift', 'threat_shift', 'goals_conceded_shift', 'position_DEF', 'position_FWD', 'position_GK', 'position_MID', 'team_Arsenal', 'team_Burnley', 'team_Man City', 'team_Newcastle', 'team_h_Other']
vif_feat_lt_20 = ['team_a_difficulty', 'bps_shift', 'minutes_shift', 'goals_scored_shift', 'goals_conceded_shift', 'clean_sheets_shift', 'bonus_shift', 'selected', 'value', 'saves_shift', 'assists_shift', 'team_h_score_shift', 'team_a_score_shift', 'yellow_cards_shift', 'penalties_saved_shift', 'round', 'penalties_missed_shift', 'red_cards_shift', 'own_goals_shift']
boruta_shap_select = ['npg_shift_per_minute', 'goals_scored_shift_per_minute', 'selected', 'minutes_shift', 'value', 'Player_Rank', 'bonus_shift', 'bonus_shift_per_minute', 'bps_shift', 'total_points_to_value', 'bps_shift_per_minute', 'assists_shift', 'bonus_shift_to_value', 'bps_shift_to_value', 'form', 'influence_shift', 'total_points_per_minute']
check = ['value', 'was_home', 'selected', 'common_transfer', 'creativity_shift',
       'minutes_shift', 'position_DMID', 'position_FWD', 'position_GK',
       'FDR_Hard', 'FDR_Low', 'position_location_General']
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# %%
# %%
GW = 1
X_train, X_test, y_train, y_test, std_x, std_y = get_training_data(GW = GW, scale = True)
# X_train, X_test = X_train[check], X_test[check]
test_scaled_net(X_train, y_train, X_test, y_test, std_y, GW)
# %%
