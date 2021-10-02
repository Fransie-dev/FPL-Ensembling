# %%
# https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
from inspect import Parameter
import sys
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import numpy as np
from fitter import Fitter, get_common_distributions
import seaborn as sns


def scale_numeric(df):
    std_scaler_X = StandardScaler()
    std_scaler_Y = StandardScaler()
    df_scaled = pd.DataFrame(std_scaler_X.fit_transform(df.drop(columns = ['total_points'], axis = 1).values), columns=df.drop(columns = ['total_points'], axis = 1).columns, index=df.index)
    df_scaled['total_points'] = std_scaler_Y.fit_transform(df['total_points'].to_numpy().reshape(-1, 1))
    return df_scaled, std_scaler_X, std_scaler_Y
    

def test_baseline(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)
    print('Baseline contains: ', X_train.shape[0], 'entries')# summarize the shape of the train and test sets
    model = LinearRegression()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    mae = mean_absolute_error(y_test, yhat)
    print('Baseline: MAE: %.3f' % mae)
    return X_train, y_train, X_test, y_test

def test_results(model, model_str, X_train, X_test, y_train, y_test, parameter = None):
    yhat = model.fit_predict(X_train)
    mask = yhat != -1
    model = LinearRegression()
    model.fit(X_train.loc[mask, :], y_train.loc[mask])
    print(f'{model_str}: removed ', X_train.shape[0] - X_train.loc[mask, :].shape[0], ' entries' )
    if parameter is not None:
        print('With ', parameter)
    yhat = model.predict(X_test)
    mae = mean_absolute_error(y_test, yhat)
    print(f'{model_str}: MAE: %.3f' % mae, end = '\n\n') 
    
    
def check_data_dist(df, feat = None):
    if feat is None:
        for feat in df.select_dtypes('number').columns:
            dist_fit = Fitter(df[feat], timeout=60*10, distributions=get_common_distributions())
            dist_fit.fit()
            key, value = list(dist_fit.get_best(method = 'sumsquare_error').items())[0] # Key is the identified distribution
            print(f'{feat} has {key} distribution')
    else:
            dist_fit = Fitter(df[feat], timeout=60*10, distributions=get_common_distributions())
            dist_fit.fit()
            dist_fit.summary() # Plots distribution 
            key, value = list(dist_fit.get_best(method = 'sumsquare_error').items())[0] # Key is the identified distribution
            print(f'{feat} has {key} distribution')

# %%
    
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Baseline
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
df = read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//cleaned_fpl.csv', index_col=0)
X, y = df.select_dtypes(include = 'number').drop(['total_points'], axis=1),df['total_points'] # split into input and output elements
X_train, y_train, X_test, y_test = test_baseline(X, y)

            
check_data_dist(df)
# %%
dist_fit = Fitter(df['total_points'], timeout=60*10, distributions=['expon', 'norm'])
dist_fit.fit()
dist_fit.summary()
# %%














# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Isolation Forest
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
for cntm in np.arange(0, 0.05, 0.01):
    iso = IsolationForest(contamination=cntm, n_estimators=100)
    test_results(iso, 'Isolation Forest',  X_train, X_test, y_train, y_test, parameter=cntm)
# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Local Outlier Factor
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
for neigh in np.arange(10, 50, 10):
    lof = LocalOutlierFactor(n_neighbors=neigh)
    test_results(lof, 'Local Outlier Factor', X_train, X_test, y_train, y_test, parameter=neigh)
# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# OneClassSVM
for nu in np.arange(0.1, 0.3, 0.1):
    osvm = OneClassSVM(nu=nu, kernel = 'rbf')
    test_results(osvm, 'One Class SVM', X_train, X_test, y_train, y_test, parameter = nu)
# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Minimum Covariance Determinant
for cntm in np.arange(0, 0.05, 0.01):
    ee = EllipticEnvelope(contamination=cntm)
    test_results(ee, 'Minimum Covariance Determinant', X_train, X_test, y_train, y_test, parameter = cntm)
# %%













# %%


# %%