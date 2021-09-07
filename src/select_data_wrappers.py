# %%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from FS.pso import jfs   # change this to switch algorithm 
import matplotlib.pyplot as plt

season = '2019-20'
training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
fpl = pd.read_csv(training_path + 'cleaned_fpl.csv', index_col=0)
# %%
# load data
data  = fpl.select_dtypes(include='number')
feat  = np.asarray(data.drop('total_points', axis=1))
label = np.asarray(data['total_points'])
# %%
# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# # parameter
k    = 5     # k-value in KNN
N    = 10    # number of chromosomes
T    = 50   # maximum number of generations
# CR   = 0.8
# MR   = 0.01
# opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'CR':CR, 'MR':MR}

c1  = 2         # cognitive factor
c2  = 2         # social factor 
w   = 0.9       # inertia weight
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'w':w, 'c1':c1, 'c2':c2}


# perform feature selection
fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']
# %%
print('Selected Features:', data.columns[fmdl['sf']].values)
print('Total Features:', len(data.columns[fmdl['sf']].values))
# %%
# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug
mdl       = KNeighborsRegressor(n_neighbors = k) 
mdl.fit(x_train, y_train)
# %%
# accuracy
import sklearn.metrics as metrics
mse=metrics.mean_squared_error(y_valid, mdl.predict(x_valid)) 
print("MSE:", mse)
# %%
# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)
# %%
# plot convergence
curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('GA')
ax.grid()
plt.show()

# %%