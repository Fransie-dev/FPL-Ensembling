# %%
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np
import pandas as pd
import statsmodels.api as sm
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import (SelectKBest, SelectPercentile,mutual_info_regression)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeRegressor
from collections import Counter
from mrmr import mrmr_regression # MRMR
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


def read_data():
    df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
    return df

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
        predominant = (data_copy[feature].value_counts() / np.float(len(data_copy))).sort_values(ascending=False).values[0]
        if predominant >= threshold:
            print(data_copy[feature].value_counts())
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

def mutual_info(X, y, select_k=0):
    if select_k == 0:
        mi = mutual_info_regression(X,y)
        mi = pd.Series(mi)
        mi.index = X.columns
        return mi.sort_values(ascending=False)

    if select_k >= 1:
        sel_ = SelectKBest(mutual_info_regression, k=select_k).fit(X,y)
        col = X.columns[sel_.get_support()]
        return col
        
    elif 0 < select_k < 1:
        sel_ = SelectPercentile(mutual_info_regression, percentile=select_k*100).fit(X,y)
        col = X.columns[sel_.get_support()]   
        return col

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


def write_log(iterations_log, data_str, elim_crit, direction): 
    log_dir = 'C://Users//jd-vz//Desktop//Code//src//features//'
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
    
def stepwise_selection(data, target,SL_in=0.05,SL_out = 0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            # print('Adding ', new_column)
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<SL_in):
            best_features.append(new_pval.idxmin())
            while(len(best_features)>0):
                best_features_with_constant = sm.add_constant(data[best_features])
                p_values = sm.OLS(target, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= SL_out):
                    excluded_feature = p_values.idxmax()
                    # print('Removing ', excluded_feature)
                    best_features.remove(excluded_feature)
                else:
                    break 
        else:
            break
    return best_features

def mi_mrmr(X_train, y_train, X_test, y_test, max_feat = 60, min_feat = 1, step_size = 1, algos = ['mrmr', 'mi']):
    ranking = pd.DataFrame(index = range(X_train.shape[1]))
    mrmr  = mrmr_regression(X_train, y_train, K = X_train.shape[1])
    ranking['mrmr'] = pd.Series([X_train.columns.get_loc(c) for c in mrmr], index = ranking.index)
    mi = mutual_info(X_train, y_train)
    ranking['mi'] = pd.Series([X_train.columns.get_loc(c) for c in mi.index], index = ranking.index)
    ks = range(min_feat, max_feat, step_size)
    loss = pd.DataFrame(index = ks, columns = algos)
    for algo in algos:
        for k in ks:
            cols = ranking[algo].sort_values().head(k).index.to_list()
            model = LinearRegression().fit(X_train.iloc[:, cols], y_train)
            loss.loc[k, algo] = metrics.mean_squared_error(y_true = y_test, y_pred = model.predict(X_test.iloc[:, cols]))
    plt.figure(figsize=(20,5))
    for algo, label, color in zip(['mrmr', 'mi'], ['MRMR', 'Mutual Info'], ['orangered', 'blue']):
            sns.pointplot(loss.index, loss[algo], label = label, color = color, lw = 3)
    plt.legend(fontsize = 13, loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.grid()
    plt.xticks([1] + list(range(min_feat, max_feat, step_size)), fontsize = 13)
    plt.xlim(-1, max_feat)
    plt.xlabel('Number of features', fontsize = 13)
    plt.ylabel('MSE', fontsize = 13)
    plt.savefig('loss.png', dpi = 300, bbox_inches = 'tight')
    return ranking, loss
    

def to_cat(df):
    for col in ['season', 'clean_sheets_shift', 'own_goals_shift', 'penalties_missed_shift', 'red_cards_shift', 'yellow_cards_shift']:
        df[col] = df[col].astype('category')
    return df

def scale_datasets(df):
  scaled_cols = []
  for col in df.columns.drop('total_points'):
    if df[col].nunique() > 2:
      scaled_cols.append(col)
  std_pred, std_resp = StandardScaler(), StandardScaler()
  X_train, y_train = df.drop('total_points', axis=1), df['total_points'] 
  X_train[scaled_cols] = pd.DataFrame(std_pred.fit_transform(X_train[scaled_cols]), 
                                      columns=df[scaled_cols].columns,
                                      index=X_train.index)
  y_train = pd.DataFrame(std_resp.fit_transform(y_train.to_numpy().reshape(-1, 1)),
                                                columns=['total_points'], index=y_train.index)
  return X_train, y_train, std_resp

def dont_scale_datasets(df):
  X_train, y_train = df.drop('total_points', axis=1), df['total_points'] 
  return X_train, y_train

def preprocess_data(df):
  # Drop indexing features
  idxrs = df[['player_name', 'kickoff_time', 'season', 'GW']]
  df = df.drop(columns = ['player_name', 'kickoff_time', 'season', 'GW'])
  # Binary encoding
  for col in df.columns:
    df[col] = df[col].replace({True:1, False:0})
  # One hot encodings
  ohe_cols = []
  for col in df.select_dtypes(include='object').columns:
      ohe_cols.append(col)
  df = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols)
  df['kickoff_time'] = idxrs['kickoff_time']
  df['GW'] = idxrs['GW']
  return df


def get_training_data(GW, scale):
    df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//engineered_us.csv')
    df.drop(['team', 'opponent_team'], inplace = True, axis = 1)
    df = preprocess_data(df)
    min_date = df[(df['GW'] == GW) & (df['kickoff_time'] >= '2020-08-12')]['kickoff_time'].min() # The first date of the gameweek
    max_date = df[(df['GW'] == GW) & (df['kickoff_time'] > '2020-08-12')]['kickoff_time'].max() # The last date of the gameweek
    df_scl = df[df['kickoff_time'] <= max_date].copy() # Only includes up to gameweek 
    std_x, std_y = StandardScaler(), StandardScaler()
    scaled_cols = []
    if scale == True:
      for col in df.columns.drop(['total_points', 'kickoff_time']):
        if df[col].nunique() > 2:
          scaled_cols.append(col)
      df_scl[scaled_cols] = std_x.fit_transform(df_scl[scaled_cols])
      df_scl['total_points'] = std_y.fit_transform(df_scl['total_points'].to_numpy().reshape(-1, 1))
      df_train, df_test = df_scl[df_scl['kickoff_time'] < min_date], df_scl[df_scl['kickoff_time'] >= min_date] 
      X = df_train.drop(['total_points', 'kickoff_time', 'GW'], axis = 1)
      y = df_train['total_points']
      X_val = df_test.drop(['total_points', 'kickoff_time', 'GW'], axis = 1)
      y_val = df_test['total_points']
    else:
      df_train, df_test = df_scl[df_scl['kickoff_time'] < min_date], df_scl[df_scl['kickoff_time'] >= min_date] 
      X = df_train.drop(['total_points', 'kickoff_time', 'GW'], axis = 1)
      y = df_train['total_points']
      X_val = df_test.drop(['total_points', 'kickoff_time', 'GW'], axis = 1)
      y_val = df_test['total_points']
    return X, X_val, y, y_val, std_x, std_y

def write_grid_results(grid, file = 'results.txt'):
  f = open(f'/content/drive/MyDrive/Parameters/{file}', 'w')
  print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
  f.write("Best: %f using %s\n\n" % (grid.best_score_, grid.best_params_))
  means = grid.cv_results_['mean_test_score']
  stds = grid.cv_results_['std_test_score']
  params = grid.cv_results_['params'] 
  used = []
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))
      f.write("%f (%f) with: %r\n\n" % (mean, stdev, param))
      used.append(param)
  f.close()
  

import seaborn as sns
sns.set()

X_train, X_test, y_train, y_test, std_x, std_y = get_training_data(GW = 3, scale = True)
ranking, loss = mi_mrmr(X_train, y_train, X_test, y_test)




# %%

X_train.shape







# %%
# TODO: Check this f_regression out
# TODO: Check RRelieFF
featureSelector = SelectKBest(score_func=f_regression,k=X_train.shape[1])
featureSelector.fit(X_train,y_train)
print([zero_based_index for zero_based_index in list(featureSelector.get_support(indices=True))])

# %%
[X_train.columns.get_loc(c) for c in list(featureSelector.get_support(indices=True))]
# %%
# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_regression, k=2)

# Apply the SelectKBest object to the features and target
X_kbest = fvalue_selector.fit_transform(X_train, y_train)

# Show results
print('Original number of features:', X_train.shape[1])
print('Reduced number of features:', X_kbest.shape[1])
# %%
# %%
    
    
    
    

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel, SelectKBest
for alph in [0.0001, 0.001, 0.01, 0.1]:
    print(alph)
    sel_=SelectFromModel(Lasso(alpha=alph))
    sel_.fit(X_train,y_train)

    sum(sel_.get_support())

    selected_feat= X_train.columns[(sel_.get_support())]

    print('total feature: {}'.format((X_train.shape[1])))
    print('selected feature: {}'.format(len(selected_feat)))
    print('dropped feature: {}'.format(np.sum(sel_.estimator_.coef_==0)))
    print(selected_feat)


# %%
# Elastic net is better in the presence of multi-collinear features
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet

pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',ElasticNet())
])
search = GridSearchCV(pipeline, {'alpha':np.arange(1e-5,1e-3,1e-1), 
                                 'l1_ratio':np.arange(0, 1, 0.01)},
                      cv = 5, scoring="neg_mean_squared_error",
                      verbose=3)

search.fit(X_train,y_train)
# %%


# %%
print(search.best_params_)
coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
# %%
# %%
print(search.best_params_)
# %%
np.array(X_train.columns)[importance > 0]
# %%
np.array(X_train.columns)[importance == 0]
# %%
print(coefficients)

# %%
importance[importance > 0]
# %%
from numpy import arange
from pandas import read_csv
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold

# define model
model = ElasticNet()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
grid['l1_ratio'] = arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X_train, y_train)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

# %%
