# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
# from sklearn.svm import SVR
import thundersvm
from thundersvm import SVR
from lr_preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
# import tensorflow
from livelossplot import PlotLossesKeras
import pandas as pd
from numba import cuda
cuda.list_devices()

# %%


def regression_results(y_true, y_pred, model_string):
    # Regression metrics
    print('# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>')
    print(model_string)
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    
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

def test_LR_model(X_train, y_train, X_test, y_test, GW):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    
    regressor = LinearRegression(normalize=True,n_jobs=-1)
    unscaled_metrics(regressor, X_train, y_train, X_test, y_test, GW, 'LR')
    
def test_SVR_model(X_train, y_train, X_test, y_test, GW):
    """[This function tests a support vector regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    regressor = SVR(C=100, epsilon=0.01, gamma=0.005, verbose=True, kernel='rbf') # Note: This was not scaled 
    unscaled_metrics(regressor, X_train, y_train, X_test, y_test, GW, 'SVR')

def test_RF_model(X_train, y_train, X_test, y_test, GW):
    """[This function tests a random forest model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    regressor = RandomForestRegressor(bootstrap=False, max_depth=60, max_features='sqrt',
                      min_samples_split=5, n_estimators=600)
    unscaled_metrics(regressor, X_train, y_train, X_test, y_test, GW, 'RF')


def unscaled_metrics(regressor, X_train, y_train, X_test, y_test, GW, model_str):
    regressor.fit(X_train, y_train)
    y_pred_test = regressor.predict(X_test)
    y_pred_test = np.round(y_pred_test) 
    y_pred_train = regressor.predict(X_train)
    y_pred_train = np.round(y_pred_train) 
    regression_results(y_test, y_pred_test, model_str)
    plot_results(y_test, y_pred_test, title = f'{model_str}: Testing, 2020 Gameweek {GW}')
    regression_results(y_train, y_pred_train, model_str)
    plot_results(y_train, y_pred_train, title = f'{model_str}: Training, 2019 + 2020 GW: {GW - 1}')

def scaled_metrics(regressor, X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW, model_str, epochs = None):
    if model_str is 'Scaled Net':
        regressor.fit(X_train, y_train, epochs=20)
    else:
        regressor.fit(X_train, y_train)
    y_train_rescaled = np.round(std_scale_Y_train.inverse_transform(y_train))
    y_test_rescaled = np.round(std_scale_Y_test.inverse_transform(y_test))
    y_pred_test = np.round(std_scale_Y_test.inverse_transform(regressor.predict(X_test)))
    y_pred_train = np.round(std_scale_Y_train.inverse_transform(regressor.predict(X_train)))
    regression_results(y_test_rescaled, y_pred_test, model_str)
    plot_results(y_test_rescaled, y_pred_test, title = f'Testing, 2020 Gameweek {GW}')
    regression_results(y_train_rescaled, y_pred_train, model_str)
    plot_results(y_train_rescaled, y_pred_train, title = f'Training, 2019 + 2020 GW: {GW - 1}')
    
def test_scaled_LR_model(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    
    regressor = LinearRegression(normalize=False,n_jobs=-1)
    scaled_metrics(regressor, X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW, 'Scaled LR')

def test_scaled_SVR_model(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    regressor = SVR(C=100, epsilon=0.01, gamma=0.005, verbose=True, kernel='rbf') # Note: This was fitted to scaled data
    scaled_metrics(regressor, X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW, 'Scaled SVR')
    
def test_scaled_RF_model(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    regressor = RandomForestRegressor(bootstrap=False, max_depth=60, max_features='sqrt',
                      min_samples_split=5, n_estimators=600)  
    scaled_metrics(regressor, X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW, 'Scaled RF')
    
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
    
def test_scaled_net(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW):
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
    regressor.add(Dense(500, activation= "relu"))
    # regressor.add(Dense(500, activation= "relu"))
    # regressor.add(Dense(500, activation= "relu")) # No diff
    regressor.add(Dense(250, activation= "relu"))
    regressor.add(Dense(1))
    regressor.compile(loss= "mse" , optimizer="adam", metrics=["mse"])
    history = regressor.fit(X_train, y_train,
          epochs=50,
          validation_split=0.2)
    y_train_rescaled = np.round(std_scale_Y_train.inverse_transform(y_train))
    y_test_rescaled = np.round(std_scale_Y_test.inverse_transform(y_test))
    y_pred_test = np.round(std_scale_Y_test.inverse_transform(regressor.predict(X_test)))
    y_pred_train = np.round(std_scale_Y_train.inverse_transform(regressor.predict(X_train)))
    regression_results(y_test_rescaled, y_pred_test, 'Neural Net')
    plot_results(y_test_rescaled, y_pred_test, title = f'Testing, 2020 Gameweek {GW}')
    regression_results(y_train_rescaled, y_pred_train, 'Neural Net')
    plot_results(y_train_rescaled, y_pred_train, title = f'Training, 2019 + 2020 GW: {GW - 1}')
    plot_history(history)
    cuda.select_device(0)
    cuda.close() # Dlt for mem

    
    
def split_data(df, df_test, GW):
    df = df.append(df_test[df_test['GW'] < GW])
    df_test = df_test[df_test['GW'] == GW] 
    X_train = df.drop(columns = ['player_name', 'GW', 'total_points']) #, 'creativity', 'ict_index', 'influence', 'threat'])
    y_train = df['total_points']
    X_test = df_test.drop(columns = ['player_name', 'GW', 'total_points']) #, 'creativity', 'ict_index', 'influence', 'threat'])
    y_test = df_test['total_points']
    return X_train, y_train, X_test, y_test

def generate_data(data_str, GW, scale, outlier_rem = False):
    if scale is True:
        df, std_scale_X_train, std_scale_Y_train, scl_feat_train = preprocess_data(data_str, '2019-20', OHE = True, FEAT = True, OUTLIER = outlier_rem, SCL = scale)
        df_test, std_scale_X_test, std_scale_Y_test, scl_feat_test  = preprocess_data(data_str, '2020-21', OHE = True, FEAT = True, OUTLIER = outlier_rem, SCL = scale)
        X_train, y_train, X_test, y_test = split_data(df, df_test, GW)
        return X_train, y_train, X_test, y_test, std_scale_X_train, std_scale_Y_train, scl_feat_train, std_scale_X_test, std_scale_Y_test, scl_feat_test
    elif scale is False:
        df = preprocess_data(data_str, '2019-20', OHE = True, FEAT = True, OUTLIER = outlier_rem, SCL = scale)
        df_test = preprocess_data(data_str, '2020-21', OHE = True, FEAT = True, OUTLIER = outlier_rem, SCL = scale)
        X_train, y_train, X_test, y_test = split_data(df, df_test, GW)
        return X_train, y_train, X_test, y_test


# %%
def test_scaled_models(data_str, GW):
    X_train, y_train, X_test, y_test, std_scale_X_train, \
        std_scale_Y_train, scl_feat_train, std_scale_X_test, \
            std_scale_Y_test, scl_feat_test = generate_data(data_str, GW = GW, scale = True, outlier_rem=False)
    # test_scaled_LR_model(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW = GW)
    test_scaled_SVR_model(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW = GW)   # Takes ~5 min
    # test_scaled_RF_model(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW = GW)  
    # test_scaled_net(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW)
    # #! SARIMAX, fireTS

def test_normal_models(GW):
    X_train, y_train, X_test, y_test = generate_data(data_str, GW =GW, scale = False)
    test_LR_model(X_train, y_train, X_test, y_test, GW = GW)
    # test_SVR_model(X_train, y_train, X_test, y_test, GW = GW) # Using scaled values
    test_RF_model(X_train, y_train, X_test, y_test, GW = GW)
# %%
# test_normal_models(GW=1)

test_scaled_models(data_str='fpl', GW = 1)
# %%
# test_scaled_models(data_str='understat', GW = 1)
# %%
# test_scaled_models(data_str='imp', GW = 1)
# %%
