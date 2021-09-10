# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import _diff_dispatcher
import pandas as pd
import seaborn as sns
import sklearn
import statsmodels.api as sm
from boruta import BorutaPy
from heatmap import corrplot, heatmap
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (SelectKBest, SelectPercentile,
                                       mutual_info_regression)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from collections import Counter

# https://github.com/runopti/stg

# %%
def read_data():
    shift = []
    # df_test = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//training//shifted_us.csv')
    # df_train = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//shifted_us.csv')
    df_test = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//training//shifted_fpl.csv')
    df_train = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//shifted_fpl.csv')
    shifted_feats = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//features//shifted.csv')['features']
    for df in df_train, df_test:
        for feat in shifted_feats:
            if feat in df.columns:
                df.drop(feat, axis=1, inplace=True)
                shift.append(feat + '_shift')
    return df_test, df_train, shift

def quasi_constant_fs(data,threshold):
    """ detect features that show the same value for the 
    majority/all of the observations (constant/quasi-constant features)
    
    Parameters
    ----------
    data : pd.Dataframe
    threshold : threshold to identify the variable as constant
        
    Returns
    -------
    list of variables names
    """
    
    data_copy = data.copy(deep=True)
    quasi_constant_features = []
    for feature in data_copy.columns:
        predominant = (data_copy[feature].value_counts() / np.float(
                      len(data_copy))).sort_values(ascending=False).values[0]
        if predominant >= threshold:
            quasi_constant_features.append(feature)
    print(len(quasi_constant_features),' variables are found to be almost constant')    
    print(quasi_constant_features)    
    return quasi_constant_features
    
    
def corr_feature_detect(data,threshold=0.8):
    """ detect highly-correlated features of a Dataframe
    Parameters
    ----------
    data : pd.Dataframe
    threshold : threshold to identify the variable correlated
        
    Returns
    -------
    pairs of correlated variables
    """
    
    corrmat = data.corr()
    corrmat = corrmat.abs().unstack() # absolute value of corr coef
    corrmat = corrmat.sort_values(ascending=False)
    corrmat = corrmat[corrmat >= threshold]
    corrmat = corrmat[corrmat < 1] # remove the digonal
    corrmat = pd.DataFrame(corrmat).reset_index()
    corrmat.columns = ['feature1', 'feature2', 'corr']
   
    grouped_feature_ls = []
    correlated_groups = []
    
    for feature in corrmat.feature1.unique():
        if feature not in grouped_feature_ls:
    
            # find all features correlated to a single feature
            correlated_block = corrmat[corrmat.feature1 == feature]
            grouped_feature_ls = grouped_feature_ls + list(
                correlated_block.feature2.unique()) + [feature]
    
            # append the block of features to the list
            correlated_groups.append(correlated_block)
    return correlated_groups


def calculate_vif(data):
    vif_df = pd.DataFrame(columns = ['Var', 'Vif'])
    x_var_names = data.columns
    for i in range(0, x_var_names.shape[0]):
        y = data[x_var_names[i]]
        x = data[x_var_names.drop([x_var_names[i]])]
        r_squared = sm.OLS(y,x).fit().rsquared
        vif = round(1/(1-r_squared),2)
        vif_df.loc[i] = [x_var_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis = 0, ascending=False, inplace=False)

def univariate_mse(df,threshold):
   
    """
    First, it builds one decision tree per feature, to predict the target
    Second, it makes predictions using the decision tree and the mentioned feature
    Third, it ranks the features according to the machine learning metric (roc-auc or mse)
    It selects the highest ranked features
    """
    df = df.select_dtypes(include='number') # NB: Only includes numeric
    X_train, X_test, y_train, y_test= train_test_split(df.drop(columns = ['total_points_shift']), df['total_points_shift'], test_size=0.2) 
    mse_values = []
    for feature in X_train.columns:
        clf = DecisionTreeRegressor(random_state=0)
        clf.fit(X_train[feature].to_frame(), y_train)
        y_scored = clf.predict(X_test[feature].to_frame())
        mse_values.append(mean_squared_error(y_test, y_scored))
    mse_values = pd.Series(mse_values)
    mse_values.index = X_train.columns
    print(mse_values.sort_values(ascending=False))
    print(len(mse_values[mse_values < threshold]),'out of the %s featues are kept'% len(X_train.columns))
    keep_col = mse_values[mse_values < threshold].sort_values(ascending=False)
    print(mse_values[mse_values > threshold].index)
    # return keep_col        


def boruta_fs(X_train, y_train):
    # The important point is for BorutaPy, multicollinearity should be removed before running it.
    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)
    feat_selector = BorutaPy(forest, verbose=2, random_state=1,max_iter = 15)
    feat_selector.fit(np.array(X_train), np.array(y_train))   # find all relevant features
    feat_selector.support_ # check selected features
    feat_selector.ranking_ # check ranking of features
    X_filtered = feat_selector.transform(np.array(X_train))    # call transform() on X to filter it down to selected features
    # zip my names, ranks, and decisions in a single iterable
    feature_ranks = list(zip(X_train.columns, 
                         feat_selector.ranking_, 
                         feat_selector.support_))

    # iterate through and print out the results
    for feat in feature_ranks:
        print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))
    return X_filtered


def write_log(iterations_log, data_str, elim_crit, direction): #TODO: incorporate me
    log_dir = 'C://Users//jd-vz//Desktop//Code//src//models//lin_reg//misc//'
    iterations_file = open(log_dir + f'{data_str}_{elim_crit}_{direction}_select_logs.txt',"w+") 
    iterations_file.write(iterations_log)
    iterations_file.close()


def backward_elimination(X, y, data_str, threshold_in = 0.05, verbose = True):
    """[This function recursively eliminates features and returns all features that are above the level of significance]

    Args:
        df ([type]): [description]
        threshold_in ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """    
    iterations_log = ""
    cols = list(X.columns)
    while (len(cols) > 0):
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y, X_1.astype(float)).fit()
        iterations_log += "\n"+str(model.summary())+"\nAIC: "+ str(model.aic) + "\nBIC: "+ str(model.bic)+"\n"
        p = pd.Series(model.pvalues.values[1:],index = cols)      
        best_pval = max(p)
        worst_feature = p.idxmax()
        if(best_pval > threshold_in):
            cols.remove(worst_feature)
            if verbose:
                print('Remove  {:30} with p-value {:.12}'.format(worst_feature, best_pval))
        else:
            break
    selected_features = cols
    print(f'{len(selected_features)} features selected')
    print(selected_features)
    write_log(iterations_log, data_str, elim_crit='t_stat', direction = 'backward')
    return selected_features


def forward_elimination(X, y, data_str, threshold_in = 0.05, verbose=True):
    """[This function recursively adds features and returns all features that are above the level of significance]

    Args:
        df ([type]): [description]
        threshold_in ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """    
    
    initial_list = []
    selected_features = list(initial_list)
    iterations_log = ""
    while True:
        changed=False
        excluded_features = list(set(X.columns)-set(selected_features))
        new_pval = pd.Series(index=excluded_features, dtype='float64')
        for new_column in excluded_features:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[selected_features+[new_column]]))).fit()
            iterations_log += "\n"+str(model.summary())+"\nAIC: "+ str(model.aic) + "\nBIC: "+ str(model.bic)+"\n"
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            selected_features.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.12}'.format(best_feature, best_pval))
        if not changed:
            break
    print(f'{len(selected_features)} features selected')
    print(selected_features)
    write_log(iterations_log, data_str, elim_crit='t_stat', direction = 'forward')
    return selected_features

        
def collect_features(be_t, fe_t):
    lst = be_t +  fe_t
    selected = pd.DataFrame(Counter(lst).most_common())
    selected = selected[selected[0] != 'intercept']
    selected.rename(columns = {0:'Feature', 1:'Occurences'}, inplace = True)
    return selected


def backward_and_forward(X, y, data_str = 'fpl'):
    log_dir = 'C://Users//jd-vz//Desktop//Code//src//features//'
    be_t = backward_elimination(X, y, data_str)
    fe_t = forward_elimination(X, y, data_str) 
    selected = collect_features(be_t, fe_t)
    selected.to_csv(log_dir + f'{data_str}_backward_and_forward.csv') 
    
    
univariate_mse(df_train, threshold=0)

# %%


df_test, df_train, shift = read_data()
for df in df_test, df_train:
    quasi_constant_features = quasi_constant_fs(df , threshold=0.99) # [['penalties_saved_shift', 'penalties_missed_shift', 'own_goals_shift', 'red_cards_shift']]
    for feat in quasi_constant_features:
        print(df[feat].value_counts())
    corr_feature_detect(df_train, threshold=0.9)
    univariate_mse(df_train, threshold=7)
    
    
vif_param = calculate_vif(X_train)
print(vif_param[vif_param['Vif'] < 15]['Var'].values.tolist())
X_filtered = boruta_fs(X_train, y_train) # generate x_train, y_train at boruta_fs
backward_and_forward(X_train, y_train, data_str='fpl')



# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from pyGRNN import GRNN
from pyGRNN import feature_selection as FS
# Loading the diabetes dataset
X = X_train
y = y_train
featnames = X_train.columns.to_list() + ['total_points_shift']
# %%

# %%
# Example 1: use Isotropic Selector to explore data dependencies in the input 
# space by analyzing the relatedness between features 
IsotropicSelector = FS.Isotropic_selector(bandwidth = 'rule-of-thumb')
IsotropicSelector.relatidness(X, feature_names = featnames)
IsotropicSelector.plot_(feature_names = featnames)
# %%
# Example 2: use Isotropic Selector to perform an exhaustive search; a rule-of-thumb
# is used to select the optimal bandwidth for each subset of features
IsotropicSelector = FS.Isotropic_selector(bandwidth = 'rule-of-thumb')
IsotropicSelector.es(X, y.ravel(), feature_names=featnames)
# %%
# Example 3: use Isotropic Selector to perform a complete analysis of the input
# space, recongising relevant, redundant, irrelevant features
IsotropicSelector = FS.Isotropic_selector(bandwidth = 'rule-of-thumb')
IsotropicSelector.feat_selection(X, y.ravel(), feature_names=featnames, strategy ='es')
# %%




# %%

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel, SelectKBest

sel_=SelectFromModel(Lasso(alpha=0.0001))
sel_.fit(X_train,y_train)

sum(sel_.get_support())

selected_feat= X_train.columns[(sel_.get_support())]

print('total feature: {}'.format((X_train.shape[1])))
print('selected feature: {}'.format(len(selected_feat)))
print('dropped feature: {}'.format(np.sum(sel_.estimator_.coef_==0)))

# %%
selected_feat
# %%

def mutual_info(X, y,select_k=10):

    if select_k >= 1:
        sel_ = SelectKBest(mutual_info_regression, k=select_k).fit(X,y)
        col = X.columns[sel_.get_support()]
    elif 0 < select_k < 1:
        sel_ = SelectPercentile(mutual_info_regression, percentile=select_k*100).fit(X,y)
        col = X.columns[sel_.get_support()]   
        
    else:
        raise ValueError("select_k must be a positive number")
    
    return col

mutual_info(X_train, y_train)
# %%


# compare different numbers of features selected using mutual information

# %%
