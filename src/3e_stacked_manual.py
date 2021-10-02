# %%
import datetime
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn.metrics as metrics
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from numpy import asarray, hstack, vstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold
import pandas as pd
import keras 
import tensorflow

num_feats = ['assists_shift',
 'bonus_shift',
 'form',
 'goals_conceded_shift',
 'bps_shift',
 'total_points_shift_to_value',
 'value',
 'bps_shift_to_value',
 'clean_sheets_shift',
 'goals_scored_shift',
 'yellow_cards_shift',
 'minutes_shift',
 'penalties_saved_shift',
 'red_cards_shift',
 'saves_shift',
 'influence_shift',
 'own_goals_shift',
 'team_Spurs',
 'transfers_out',
 'team_Brighton',
 'total_points_shift_per_minute',
 'bps_shift_per_minute',
 'top_scorer',
 'team_Man City',
 'minutes_shift_cumulative',
 'team_Wolves',
 'team_Newcastle',
 'xGChain_shift',
 'transfers_balance',
 'npxG_shift',
 'xGBuildup_shift',
 'matches_shift_cumulative',
 'opponent_team_Brighton',
 'xG_shift']


def plot_results(true_value, predicted_value, title):
    plt.figure(figsize=(10, 10))
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
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))

def baseline_model():
    model = Sequential()
    model.add(Dense(500, activation= "relu",))
    model.add(Dense(500, activation= "relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
step_feat = ['goals_conceded_shift','npg_shift','form','bonus_shift','bps_shift','clean_sheets_shift','assists_shift','total_points_shift_to_value','npg_shift_to_value','influence_shift_to_value','value','bps_shift_to_value','bonus_shift_to_value','influence_shift','yellow_cards_shift','goals_scored_shift','goals_scored_shift_to_value','minutes_shift','penalties_saved_shift','saves_shift','own_goals_shift','red_cards_shift','position_DEF','transfers_out','xGChain_shift','Player_Rank_No','total_points_shift_per_minute','bps_shift_per_minute','team_Spurs','team_Wolves','team_Chelsea','bonus_shift_per_minute','opponent_team_Brighton','xA_shift','Supporters','team_Newcastle','team_Man Utd','team_Everton','team_Brighton','opponent_team_West Ham','threat_shift','transfers_balance','position_GK']
get_step_data = FunctionTransformer(lambda x: x[step_feat], validate=False)

def get_models():
    models = list()
    models.append(('SW', Pipeline([('selector', get_step_data), ('LR', LinearRegression())])))   
    # models.append(SVR(C=100, epsilon=0.01, kernel='rbf'))
    # models.append(KNeighborsRegressor(n_neighbors=3))
    # models.append(RandomForestRegressor())
    # models.append(KerasRegressor(build_fn=baseline_model, epochs=50, verbose=False,validation_split=0.2))
    return models

def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = list(), list()
    # define split of data
    kfold = KFold(n_splits=5, shuffle=True)
    # enumerate splits
    for train_ix, test_ix in kfold.split(X):
        fold_yhats = list()
        # get data
        train_X, test_X = X.iloc[train_ix, :], X.iloc[test_ix, :]
        train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
        meta_y.extend(test_y)
        # fit and make predictions with each sub-model
        for model in models:
            # print('Training', model.__class__.__name__) # TODO: GET FEAT
            # print('Split', model.__class__.__name__) # TODO: GET FEAT
            model.fit(train_X, train_y)  # TODO: Train on model feat
            yhat = model.predict(test_X)  # TODO: Predict on model feat
            fold_yhats.append(yhat.reshape(len(yhat), 1)) # store columns
        meta_X.append(hstack(fold_yhats))
    return vstack(meta_X), asarray(meta_y)

def fit_base_models(X, y, models):
    for model in models:
        model.fit(X, y)
    return models

def fit_meta_model(X, y):
    # model = KerasRegressor(build_fn=baseline_model, epochs=50,verbose=False, validation_split=0.2)
    model = LinearRegression()
    model.fit(X, y)
    return model

def evaluate_models(X, y, models, std_y):
    for model in models:
        yhat = np.round(std_y.inverse_transform(model.predict(X)))
        y = np.round(std_y.inverse_transform(y))
        mse = mean_squared_error(y, yhat)
        print('%s: RMSE %.3f' % (model.__class__.__name__, sqrt(mse)))

def super_learner_predictions(X, models, meta_model):
    meta_X = list()
    for model in models:
        yhat = model.predict(X)
        meta_X.append(yhat.reshape(len(yhat),1))
    meta_X = hstack(meta_X)
    # predict
    return meta_model.predict(meta_X)

def get_training_data(GW, num_feats):
    df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
    min_date = df[(df['GW'] == GW) & (df['kickoff_time'] >= '2020-08-12')]['kickoff_time'].min() # The first date of the gameweek
    max_date = df[(df['GW'] == GW) & (df['kickoff_time'] > '2020-08-12')]['kickoff_time'].max() # The last date of the gameweek
    df_scl = df[df['kickoff_time'] <= max_date].copy() # Only includes up to gameweek 
    std_x, std_y = StandardScaler(), StandardScaler()
    df_scl[num_feats] = std_x.fit_transform(df_scl[num_feats])
    df_scl['total_points_shift'] = std_y.fit_transform(df_scl['total_points_shift'].to_numpy().reshape(-1, 1))
    df_train, df_test = df_scl[df_scl['kickoff_time'] < min_date], df_scl[df_scl['kickoff_time'] >= min_date] 
    X = df_train[num_feats] 
    y = df_train['total_points_shift']
    X_val = df_test[num_feats] 
    y_val = df_test['total_points_shift']
    return X, X_val, y, y_val, std_x, std_y, df_test

def get_stacked_mod(GW, num_feats):
    X, X_val, y, y_val, std_x, std_y, df_test = get_training_data(GW, num_feats)
    print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)
    models = get_models()
    meta_X, meta_y = get_out_of_fold_predictions(X, y, models)
    print('Meta ', meta_X.shape, meta_y.shape)
    models = fit_base_models(X, y, models)
    meta_model = fit_meta_model(meta_X, meta_y)
    evaluate_models(X_val, y_val, models, std_y)
    yhat = super_learner_predictions(X_val, models, meta_model)
    # regression_results(np.round(std_y.inverse_transform(y_val)), np.round(std_y.inverse_transform(yhat)), 'Stacked')
    # plot_results(y_val, yhat, title = f'Testing Stacking for GW ')
    print('Super Learner: RMSE %.3f' % (sqrt(mean_squared_error(np.round(std_y.inverse_transform(y_val)), np.round(std_y.inverse_transform(yhat))))))
    # print(pd.DataFrame({'True': np.round(std_y.inverse_transform(y_val)), 'Predicted': np.round(std_y.inverse_transform(yhat))}))
    return np.round(std_y.inverse_transform(yhat)).flatten().tolist()

def collect_results():
    results = []
    for gw in range(1, 39):
        print('Gameweek', gw)
        results.append(get_stacked_mod(gw, num_feats))
    df_orig = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_us.csv')
    df_eng = pd.read_csv('C://Users//jd-vz//Desktop//Code//data/engineered_us.csv')
    df = df_orig[['player_name', 'position', 'team', 'GW', 'value', 'kickoff_time']].copy()
    df = df[df['kickoff_time'] > '2020-08-12']
    df['total_points'] = df_eng['total_points_shift']
    df['predicted_points'] = [item for subl in results for item in subl]
    df.to_csv('C://Users//jd-vz//Desktop//Code//data//stacked_predictions.csv', index=False) 
    # TODO: Uncertain about ordering of predictions
# %%

collect_results()
# %%
# df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//stacked_predictions.csv')
# %%
# import seaborn as sns
# df_pos = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_us.csv')
# df_eng = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
# df_eng = df_eng[(df_eng['kickoff_time'] >= min_date) & (df_eng['kickoff_time']  <= max_date)]
# df_pos = df_pos[(df_pos['kickoff_time'] >= min_date) & (df_pos['kickoff_time']  <= max_date)]
# df_eng['pred_total_points'] = np.round(std_y.inverse_transform(yhat))
# df_eng['position'] = df_pos['position'].values# %%
# sns.lmplot(x="pred_total_points", y="total_points_shift", hue="position", data=df_eng)
