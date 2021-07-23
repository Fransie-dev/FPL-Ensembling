# %%

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
from sklearn.svm import SVR
from lr_preprocess import preprocess_data
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from lr_preprocess import preprocess_data

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

def rand_rand_search():
    X_train, y_train, X_test, y_test = generate_data(GW =1, scale = False)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = GridSearchCV()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train) # 12h00
    rf_random.best_estimator_
    # regressor = RandomForestRegressor(bootstrap=False, max_depth=60, max_features='sqrt',
                    #   min_samples_split=5, n_estimators=600) # Best model


# %%
X_train, y_train, X_test, y_test = generate_data(GW =1, scale = False)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 500, stop = 700, num = 5)]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(50, 70, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [5]
# Minimum number of samples required at each leaf node
# Method of selecting samples for training each tree
bootstrap = [False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_grid = GridSearchCV(estimator = rf, param_grid= random_grid, cv=3, verbose=2)
# Fit the random search model
rf_grid.fit(X_train, y_train) # 12h00
rf_grid.best_estimator_
# %%
regressor = RandomForestRegressor(bootstrap=False, max_depth=60, max_features='sqrt',
                    min_samples_split=5, n_estimators=600)
regressor.fit(X_train, y_train)
y_pred_test = regressor.predict(X_test)
y_pred_test = np.round(y_pred_test) # NB: Rounding
y_pred_train = regressor.predict(X_train)
y_pred_train = np.round(y_pred_train) # NB: Rounding
# %%
regression_results(y_test, y_pred_test, 'RF')
# %%
