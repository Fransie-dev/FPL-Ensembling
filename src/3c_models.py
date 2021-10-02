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
from sklearnex import patch_sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
patch_sklearn()

def read_data():
    df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
    return df_test, df_train, shift

def change_different_teams(df_test, df_train):
    changed_teams = ['Fulham', 'Leeds', 'West Brom', 'Watford', 'Bournemouth', 'Norwich'] # These teams are not in both seasons
    for df in df_test, df_train:
        for feat in ['team', 'opponent_team', 'team_a', 'team_h']:
            df.loc[df[feat].isin(changed_teams), feat] = 'Other'
            df[feat].astype("category")
    return df_test, df_train


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
    regressor.add(Dense(96, activation= "relu",))
    regressor.add(Dense(416, activation= "relu"))
    regressor.add(Dense(1))
    regressor.compile(loss= "mse" , optimizer="adam", metrics=["mse"])
    history = regressor.fit(X_train, y_train, epochs=20, validation_split=0.2)
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
 
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Features
univariate_mse_remove = ['selected', 'transfers_in_shift', 'transfers_out_shift', 'transfers_balance_shift']
quasi_remove = ['penalties_saved_shift', 'penalties_missed_shift', 'own_goals_shift', 'red_cards_shift'] 
boruta_select= ['round' ,'value' ,'clean_sheets_shift','goals_scored_shift','saves_shift','assists_shift','bps_shift','influence_shift' ,'yellow_cards_shift' ,'creativity_shift','own_goals_shift'  ,       'red_cards_shift' ,        'minutes_shift' ,          'bonus_shift'   ,          'ict_index_shift' ,        'threat_shift' ,           'goals_conceded_shift',    'position_DEF' ,           'position_FWD',            'position_GK'  ,           'position_MID']        
fwd_select = ['threat_shift', 'bonus_shift', 'creativity_shift', 'influence_shift', 'minutes_shift', 'goals_conceded_shift', 'clean_sheets_shift', 'assists_shift', 'goals_scored_shift', 'yellow_cards_shift', 'bps_shift', 'penalties_saved_shift', 'saves_shift', 'red_cards_shift', 'own_goals_shift', 'position_MID', 'ict_index_shift', 'penalties_missed_shift', 'position_DEF', 'strength_defence_home', 'value', 'team_Man City', 'team_Newcastle', 'team_Arsenal', 'team_Chelsea', 'team_h_Sheffield Utd']
bwd_select = ['value', 'strength_defence_home', 'clean_sheets_shift', 'goals_scored_shift', 'penalties_saved_shift', 'saves_shift', 'assists_shift', 'bps_shift', 'influence_shift', 'penalties_missed_shift', 'yellow_cards_shift', 'creativity_shift', 'own_goals_shift', 'red_cards_shift', 'minutes_shift', 'bonus_shift', 'ict_index_shift', 'threat_shift', 'goals_conceded_shift', 'position_DEF', 'position_FWD', 'position_GK', 'position_MID', 'team_Arsenal', 'team_Burnley', 'team_Man City', 'team_Newcastle', 'team_h_Other']
vif_feat_lt_20 = ['team_a_difficulty', 'bps_shift', 'minutes_shift', 'goals_scored_shift', 'goals_conceded_shift', 'clean_sheets_shift', 'bonus_shift', 'selected', 'value', 'saves_shift', 'assists_shift', 'team_h_score_shift', 'team_a_score_shift', 'yellow_cards_shift', 'penalties_saved_shift', 'round', 'penalties_missed_shift', 'red_cards_shift', 'own_goals_shift']
boruta_shap_select = ['npg_shift_per_minute', 'goals_scored_shift_per_minute', 'selected', 'minutes_shift', 'value', 'Player_Rank', 'bonus_shift', 'bonus_shift_per_minute', 'bps_shift', 'total_points_shift_to_value', 'bps_shift_per_minute', 'assists_shift', 'bonus_shift_to_value', 'bps_shift_to_value', 'form', 'influence_shift', 'total_points_shift_per_minute']
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# %%
# %%
df_test, df_train, shift = read_data()
# df_test, df_train = change_different_teams(df_test, df_train)
GW = 3
min_date = df_test[df_test['GW'] == GW]['kickoff_time'].min() # The first date of the gameweek
df_scl = df_train.append(df_test[df_test['GW'] <= GW]) # Only includes up to gameweek 
num_feats = df_scl.select_dtypes(include='number').drop(columns = ['total_points_shift', 'GW']).columns
extracts = ['player_name', 'GW', 'total_points_shift', 'kickoff_time']
std_scaler_X, std_scaler_Y = StandardScaler(), StandardScaler()
df_scl[num_feats] = std_scaler_X.fit_transform(df_scl[num_feats])
df_scl['total_points_shift'] = std_scaler_Y.fit_transform(df_scl['total_points_shift'].to_numpy().reshape(-1, 1))
df_scl = one_hot_encode(df_scl)
df_train, df_test = df_scl[df_scl['kickoff_time'] < min_date], df_scl[df_scl['kickoff_time'] >= min_date] 
X_train = df_train.drop(columns = extracts) 
y_train = df_train['total_points_shift']
X_test = df_test.drop(columns = extracts) 
y_test = df_test['total_points_shift']
select = boruta_shap_select
X_train, X_test = X_train[select], X_test[select]
test_scaled_net(X_train, y_train, X_test, y_test, std_scaler_Y, GW)
# %%
# %%
# test_scaled_RF_model(X_train, y_train, X_test, y_test, std_scaler_Y, GW)

# %%
# from numpy import mean
# from numpy import std
# from sklearn.datasets import make_regression
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedKFold
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import mutual_info_regression
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# from matplotlib import pyplot
# # define dataset
# X, y = X_train, y_train
# # define number of features to evaluate
# num_features = [i for i in range(X.shape[1]-19, X.shape[1]+1)]
# # enumerate each number of features
# results = list()
# for k in num_features:
# 	# create pipeline
# 	model = LinearRegression()
# 	fs = SelectKBest(score_func=mutual_info_regression, k=k)
# 	pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
# 	# evaluate the model
# 	cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
# 	scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# 	results.append(scores)
# 	# summarize the results
# 	print('>%d %.3f (%.3f)' % (k, mean(scores), std(scores)))
# # plot model performance for comparison
# pyplot.boxplot(results, labels=num_features, showmeans=True)
# pyplot.show()
# %%

# select = fwd_select
# X_train, X_test = X_train[select], X_test[select]

# test_scaled_LR_model(X_train, y_train, X_test, y_test, std_scaler_Y, GW)
# test_scaled_SVR_model(X_train, y_train, X_test, y_test, std_scaler_Y, GW) 
# test_scaled_knn_model(X_train, y_train, X_test, y_test, std_scaler_Y, GW) 
# test_scaled_net(X_train, y_train, X_test, y_test, std_scaler_Y, GW)
# test_scaled_RF_model(X_train, y_train, X_test, y_test, std_scaler_Y, GW)






def reduce_multicollin(X_train, y_train, X_test, y_test):
    pca = PCA(n_components = X_train.shape[1])
    pca_data = pca.fit_transform(X_train)
    percent_var_explained = pca.explained_variance_/(np.sum(pca.explained_variance_))
    cumm_var_explained = np.cumsum(percent_var_explained)
    plt.plot(cumm_var_explained)
    plt.grid()
    plt.xlabel("n_components")
    plt.ylabel("% variance explained")
    plt.show()
    sum(pca.explained_variance_ratio_)
    # for comp in range(50,60,1):
    comp = 52
    pca = PCA(n_components=comp)
    pca_train_data = pca.fit_transform(X_train)
    pca_test_data = pca.transform(X_test)
    print(comp, end = '\n\n')
    test_scaled_LR_model(pca_train_data, y_train, pca_test_data, y_test, std_scaler_Y, GW)
    # df_train_pca = pd.DataFrame(pca_train_data)
    # df_train_pca["total_points_shift"] = y_train.values
    # corr = df_train_pca.corr()
    # plt.figure(figsize = (12,10))
    # sns.heatmap(corr, annot = True, vmin=-1, vmax=1, cmap="YlGnBu", linewidths=.5)
    # plt.grid(b=True, color='#f68c1f', alpha=0.1)
    # plt.show()
    




reduce_multicollin(X_train, y_train, X_test, y_test)
test_scaled_LR_model(X_train, y_train, X_test, y_test, std_scaler_Y, GW)
# %%