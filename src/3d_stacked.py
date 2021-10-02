# %%
from math import sqrt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from mlens.ensemble import SuperLearner
from mlens.preprocessing import Subset
from sklearn.preprocessing import StandardScaler
# create a list of base-models
def get_models():
    models = list()
    models.append(LinearRegression())
    models.append(SVR())
    models.append(KNeighborsRegressor())
    models.append(RandomForestRegressor(n_estimators=10))
    return models

# cost function for base models
def rmse(yreal, yhat):
    return sqrt(mean_squared_error(yreal, yhat))

# create the super learner
def get_super_learner(X):
    # preprocessing = {'pipeline-1': [StandardScaler(), Subset([0,1])], 
    preprocessing = {'pipeline-1': [Subset([5,6])], 
                     'pipeline-2': [Subset([5,6])]}
    estimators = {'pipeline-1': [LinearRegression()], 
                  'pipeline-2': [LinearRegression()]}
    ensemble = SuperLearner(scorer=rmse, folds=10, shuffle=True, sample_size=len(X))
    ensemble.add(estimators, preprocessing)
    ensemble.add_meta(LinearRegression()) 
    return ensemble
    # add base models
    # models = get_models()
    # ensemble.add(models)
    # add the meta model

# create the inputs and outputs

X, y = make_regression(n_samples=1000, n_features=100, noise=0.5)
# split
X, X_val, y, y_val = train_test_split(X, y, test_size=0.50)
print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)
# create the super learner
ensemble = get_super_learner(X)
# fit the super learner
ensemble.fit(X, y)
# %%
# summarize base learners
print(ensemble.data)
# %%
# evaluate meta model
yhat = ensemble.predict(X_val)
print('Super Learner: RMSE %.3f' % (rmse(y_val, yhat)))
# %%
ensemble.data
# %%
ensemble.layers
# %%
