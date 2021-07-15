# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.svm import SVR
from lr_preprocess import preprocess_data


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
    regressor.fit(X_train, y_train)
    y_pred_test = regressor.predict(X_test)
    y_pred_test = np.round(y_pred_test) # NB: Rounding
    y_pred_train = regressor.predict(X_train)
    y_pred_train = np.round(y_pred_train) # NB: Rounding
    regression_results(y_test, y_pred_test, 'Linear Regression')
    plot_results(y_test, y_pred_test, title = f'Testing, 2020 Gameweek {GW}')
    regression_results(y_train, y_pred_train, 'Linear Regression')
    plot_results(y_train, y_pred_train, title = f'Training, 2019 + 2020 GW: {GW - 1}')

def test_scaled_LR_model(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    
    regressor = LinearRegression(normalize=False,n_jobs=-1)
    regressor.fit(X_train, y_train)
    y_tr = np.round(std_scale_Y_train.inverse_transform(y_train))
    y_te = np.round(std_scale_Y_test.inverse_transform(y_test))
    y_pred_test = np.round(std_scale_Y_test.inverse_transform(regressor.predict(X_test)))
    y_pred_train = np.round(std_scale_Y_train.inverse_transform(regressor.predict(X_train)))
    regression_results(y_te, y_pred_test, 'Scaled Linear Regression')
    plot_results(y_te, y_pred_test, title = f'Testing, 2020 Gameweek {GW}')
    regression_results(y_tr, y_pred_train, 'Scaled Linear Regression')
    plot_results(y_tr, y_pred_train, title = f'Training, 2019 + 2020 GW: {GW - 1}')

def test_scaled_SVR_model(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    regressor = SVR(C=100, epsilon=0.01, gamma=0.005, verbose=True, kernel='rbf')#** pycaret, sklearn cross validation
    regressor.fit(X_train, y_train)
    y_tr = np.round(std_scale_Y_train.inverse_transform(y_train))
    y_te = np.round(std_scale_Y_test.inverse_transform(y_test))
    y_pred_test = np.round(std_scale_Y_test.inverse_transform(regressor.predict(X_test)))
    y_pred_train = np.round(std_scale_Y_train.inverse_transform(regressor.predict(X_train)))
    regression_results(y_te, y_pred_test, 'Scaled SVR')
    plot_results(y_te, y_pred_test, title = f'Testing, 2020 Gameweek {GW}')
    regression_results(y_tr, y_pred_train, 'Scaled SVR')
    plot_results(y_tr, y_pred_train, title = f'Training, 2019 + 2020 GW: {GW - 1}')
    
    
def test_SVR_model(X_train, y_train, X_test, y_test, GW):
    """[This function tests a support vector regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    regressor = SVR(C=100, epsilon=0.01, gamma=0.005, verbose=True, kernel='rbf') # Scaled SVR parameters 
    regressor.fit(X_train, y_train)
    y_pred_test = regressor.predict(X_test)
    y_pred_test = np.round(y_pred_test) # NB: Rounding
    y_pred_train = regressor.predict(X_train)
    y_pred_train = np.round(y_pred_train) # NB: Rounding
    regression_results(y_test, y_pred_test, 'SVR')
    plot_results(y_test, y_pred_test, title = f'SVR: Testing, 2020 Gameweek {GW}')
    regression_results(y_train, y_pred_train, 'SVR')
    plot_results(y_train, y_pred_train, title = f'SVR: Training, 2019 + 2020 GW: {GW - 1}')

    
def split_data(df, df_test, GW):
    df = df.append(df_test[df_test['GW'] < GW])
    df_test = df_test[df_test['GW'] == GW] 
    df = df.drop(columns = ['player_name', 'GW'])
    df_test = df_test.drop(columns = ['player_name', 'GW'])
    X_train = df.drop(columns = ['total_points'])
    y_train = df['total_points']
    X_test = df_test.drop(columns = ['total_points'])
    y_test = df_test['total_points']
    return X_train, y_train, X_test, y_test

def generate_data(GW, scale, outlier_rem = False):
    if scale is True:
        df, std_scale_X_train, std_scale_Y_train, scl_feat_train = preprocess_data('fpl', '2019-20', OHE = True, FEAT = True, OUTLIER = outlier_rem, SCL = scale)
        df_test, std_scale_X_test, std_scale_Y_test, scl_feat_test  = preprocess_data('fpl', '2020-21', OHE = True, FEAT = True, OUTLIER = outlier_rem, SCL = scale)
        X_train, y_train, X_test, y_test = split_data(df, df_test, GW)
        return X_train, y_train, X_test, y_test, std_scale_X_train, std_scale_Y_train, scl_feat_train, std_scale_X_test, std_scale_Y_test, scl_feat_test
    elif scale is False:
        df = preprocess_data('fpl', '2019-20', OHE = True, FEAT = True, OUTLIER = outlier_rem, SCL = scale)
        df_test = preprocess_data('fpl', '2020-21', OHE = True, FEAT = True, OUTLIER = outlier_rem, SCL = scale)
        X_train, y_train, X_test, y_test = split_data(df, df_test, GW)
        return X_train, y_train, X_test, y_test


# %%
def test_scaled_models(GW):
    X_train, y_train, X_test, y_test, std_scale_X_train, \
        std_scale_Y_train, scl_feat_train, std_scale_X_test, \
            std_scale_Y_test, scl_feat_test = generate_data(GW = GW, scale = True, outlier_rem=False)
    test_scaled_LR_model(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW = GW)
    test_scaled_SVR_model(X_train, y_train, X_test, y_test, std_scale_Y_train, std_scale_Y_test, GW = GW)   # Takes ~5 min
    # #! SARIMAX, fireTS

def test_normal_models(GW):
    X_train, y_train, X_test, y_test = generate_data(GW =GW, scale = False)
    test_LR_model(X_train, y_train, X_test, y_test, GW = GW)
    test_SVR_model(X_train, y_train, X_test, y_test, GW = GW)
# %%
test_normal_models(GW=1)

# %%
test_scaled_models(GW = 1)
# %%
