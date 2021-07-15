from sklearn.model_selection import RandomizedSearchCV
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
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features, train_labels)