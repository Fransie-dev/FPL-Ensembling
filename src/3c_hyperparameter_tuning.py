# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
import sklearn
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
import joblib
# %%
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
df = df[df['kickoff_time'] <= '2020-07-26'] # Only consider 2019/20 season for hyperparameter tuning
df.drop(['player_name', 'GW', 'kickoff_time'], axis=1, inplace=True)
std_pred, std_resp = StandardScaler(), StandardScaler()
X_train, y_train = df.drop('total_points_shift', axis=1), df['total_points_shift']
X_train, y_train = X_train.loc[0:200], y_train.loc[0:200]


from sklearn.model_selection import RandomizedSearchCV
rf_n_estimators = [int(x) for x in np.linspace(200, 1000, 5)]
rf_n_estimators.append(1500)
rf_n_estimators.append(2000)
rf_max_depth = [int(x) for x in np.linspace(5, 55, 11)]
rf_max_depth.append(None)
rf_max_features = ['auto', 'sqrt', 'log2']
rf_criterion = ['mse', 'mae']
rf_min_samples_split = [int(x) for x in np.linspace(2, 10, 9)]
rf_min_impurity_decrease = [0.0, 0.05, 0.1]
rf_bootstrap = [True, False]
rf_grid = {'n_estimators': rf_n_estimators,
               'max_depth': rf_max_depth,
               'max_features': rf_max_features,
               'criterion': rf_criterion,
               'min_samples_split': rf_min_samples_split,
               'min_impurity_decrease': rf_min_impurity_decrease,
               'bootstrap': rf_bootstrap}
rf_base = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf_base, param_distributions = rf_grid, 
                               n_iter = 2, cv = 3, verbose = 2, random_state = 42, 
                               n_jobs = 4)
rf_random.fit(X_train, y_train)
joblib.dump(rf_random, 'random_forst_search.pkl')
# %%

























# Obtain final model performance
def cv_comparison(models, X, y, cv):
    avg_results, detailed_results = pd.DataFrame(), pd.DataFrame(index=range(1, cv + 1))
    for model in models:
        mae = -np.round(cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv), 4)
        mae_avg = round(mae.mean(), 4)
        mse = -np.round(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv), 4)
        mse_avg = round(mse.mean(), 4)
        r2 = np.round(cross_val_score(model, X, y, scoring='r2', cv=cv), 4)
        r2_avg = round(r2.mean(), 4)
        avg_results[str(model)]  = [mae_avg, mse_avg, r2_avg]
        detailed_results[model.__class__.__name__ + 'MAE'] = mae
        detailed_results[model.__class__.__name__ + 'MSE'] = mse
        detailed_results[model.__class__.__name__ + 'R2'] = r2
    avg_results.index = ['Mean Absolute Error', 'Mean Squared Error', 'R^2']
    return avg_results, detailed_results

mlr_reg = LinearRegression()
rf_reg = LinearRegression(fit_intercept=False)

models = [mlr_reg, rf_reg]
# Run the Cross-Validation comparison 
avg_results, detailed_results = cv_comparison(models, X_train, y_train, cv=4)

# %%

# %%
# %%
