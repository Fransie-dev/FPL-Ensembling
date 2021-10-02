# %%
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import sys
sys.path.insert(0, 'C://Users//jd-vz//Desktop//Code//src//models//lin_reg//')
from lr_model import generate_data
# %%



def svc_param_selection(X, y, nfolds):
    # Poly + RBF
    Cs = [0.1, 1, 10, 100]
    gammas = [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
    epsilons = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    param_grid = {'C': Cs, 'gamma' : gammas, 'epsilon' : epsilons}
    grid_search = GridSearchCV(SVR(verbose=3), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    # grid_search.best_params_
    return grid_search

# X_train, y_train, X_test, y_test = generate_data(GW = 1, scale = False, outlier_rem=False)
X_train, y_train, X_test, y_test, std_scale_X_train, \
    std_scale_Y_train, scl_feat_train, std_scale_X_test, \
        std_scale_Y_test, scl_feat_test = generate_data(GW = 1, scale = True, outlier_rem=False)
# %%
# %% time # 3hr 40 min
# 14:46
grid_search = svc_param_selection(X_train, y_train, nfolds=2) # 10 folds is too much. To per parameter

# %%
grid_search.best_params_
# %%
grid_search.best_estimator_
# %%
grid_search.best_score_
# %%
from sklearn.neighbors import KNeighborsRegressor
