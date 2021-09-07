# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from boruta import BorutaPy
from heatmap import corrplot, heatmap
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (SelectKBest, SelectPercentile,
                                       VarianceThreshold, chi2,
                                       mutual_info_classif)
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from utilities import one_hot_encode, split_data_by_GW
from sklearn.model_selection import train_test_split
import infoselect as inf


def read_data(season):
    """[This function reads the data]

    Args:
        season ([type]): [description]

    Returns:
        [type]: [description]
    """    
    training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
    fpl = pd.read_csv(training_path + 'cleaned_fpl.csv', index_col=0)
    understat = pd.read_csv(training_path + 'cleaned_understat.csv', index_col=0)
    return fpl, understat

def quasi_constant_fs(data,threshold=0.99):
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
    quasi_constant_feature = []
    for feature in data_copy.columns:
        predominant = (data_copy[feature].value_counts() / np.float(
                      len(data_copy))).sort_values(ascending=False).values[0]
        if predominant >= threshold:
            quasi_constant_feature.append(feature)
    print(len(quasi_constant_feature),' variables are found to be almost constant')    
    return quasi_constant_feature

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

def mutual_info(df,select_k=10):
    df = df.select_dtypes(include='number') # NB: Only includes numeric
    X = df.drop(columns = ['total_points'])
    y = df['total_points']
    if select_k >= 1:
        sel_ = SelectKBest(mutual_info_classif, k=select_k).fit(X,y)
        col = X.columns[sel_.get_support()]
    elif 0 < select_k < 1:
        sel_ = SelectPercentile(mutual_info_classif, percentile=select_k*100).fit(X,y)
        col = X.columns[sel_.get_support()]   
        
    else:
        raise ValueError("select_k must be a positive number")
    
    return col

def univariate_mse(df,threshold):
   
    """
    First, it builds one decision tree per feature, to predict the target
    Second, it makes predictions using the decision tree and the mentioned feature
    Third, it ranks the features according to the machine learning metric (roc-auc or mse)
    It selects the highest ranked features
    """
    df = df.select_dtypes(include='number') # NB: Only includes numeric
    X_train, X_test, y_train, y_test= train_test_split(df.drop(columns = ['total_points']), df['total_points'], test_size=0.2) 
    mse_values = []
    for feature in X_train.columns:
        clf = DecisionTreeRegressor()
        clf.fit(X_train[feature].to_frame(), y_train)
        y_scored = clf.predict(X_test[feature].to_frame())
        mse_values.append(mean_squared_error(y_test, y_scored))
    mse_values = pd.Series(mse_values)
    mse_values.index = X_train.columns
    print(mse_values.sort_values(ascending=False))
    print(len(mse_values[mse_values > threshold]),'out of the %s featues are kept'% len(X_train.columns))
    keep_col = mse_values[mse_values > threshold]
    return keep_col        


fpl, understat = read_data('2019-20')
quasi_constant_fs(fpl, threshold=0.99) # Check for quasi-constant features in the data set..
corr_feature_detect(fpl, threshold=0.95)
# %%
univariate_mse(fpl, threshold=0)
# %%
mutual_info(fpl,select_k=33)
# %%



















# %%
def remove_multicoll(df):
    """[This function removes columns with ]

    Args:
        df ([type]): [description]
    """    
    df = df.select_dtypes(include='number')
    plt.figure(figsize=(20,20))
    corrplot(df.corr(), size_scale=700, marker='s')

def PLS_regression(data):
    """[For removing multicollinearityt]

    Args:
        data ([type]): [description]
    """    
    #define predictor and response variables
    X = data.select_dtypes(include='number').drop('total_points', axis = 1)
    y = data['total_points']
    #define cross-validation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    mse = []
    n = len(X)
    # Calculate MSE with only the intercept
    score = -1*model_selection.cross_val_score(PLSRegression(n_components=1),
            np.ones((n,1)), y, cv=cv, scoring='neg_mean_squared_error').mean()    
    mse.append(score)
    # Calculate MSE using cross-validation, adding one component at a time
    for i in np.arange(1, len(data.columns) + 1):
        pls = PLSRegression(n_components=i)
        score = -1*model_selection.cross_val_score(pls, scale(X), y, cv=cv,
                scoring='neg_mean_squared_error').mean()
        mse.append(score)
    #plot test MSE vs. number of components
    plt.plot(mse)
    plt.xlabel('Number of PLS Components')
    plt.ylabel('MSE')
    plt.title('total_points')

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


PLS_regression(fpl)
# remove_multicoll(fpl)
# calculate_vif(fpl.select_dtypes(include='number').drop('total_points', axis = 1))
# calculate_vif(fpl.select_dtypes(include='number').drop(['total_points', 'team_a_strength_overall', 'team_h_strength_defense', 'team_h_strength_attack', 'team_a_strength_attack', 'team_a_strength_defense', 'team_h_strength_overall'], axis = 1))

# %%
def boruta_fs(df, df_test):
    # The important point is for BorutaPy, multicollinearity should be removed before running it.
    X_train, y_train, X_test, y_test = split_data_by_GW(df, df_test) # Where X_train and y_train contain the entire season
    forest = RandomForestRegressor(n_jobs=-1,  max_depth=5)
    forest.fit(X_train, y_train)
    feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
    feat_selector.fit(np.array(X_train), np.array(y_train))   # find all relevant features
    feat_selector.support_ # check selected features
    feat_selector.ranking_ # check ranking of features
    X_filtered = feat_selector.transform(X_train)    # call transform() on X to filter it down to selected features
    # zip my names, ranks, and decisions in a single iterable
    feature_ranks = list(zip(X_train.columns, 
                         feat_selector.ranking_, 
                         feat_selector.support_))
    # iterate through and print out the results
    for feat in feature_ranks:
        print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

fpl_19, understat_19 = read_data('2019-20')
fpl_20, understat_20 = read_data('2020-21')



# %%
fpl_19 = one_hot_encode(fpl_19.drop('kickoff_time', axis = 1)).reset_index(drop=True)
fpl_20 = one_hot_encode(fpl_20.drop('kickoff_time', axis = 1)).reset_index(drop = True)
boruta_fs(clean_dataset(fpl_19), clean_dataset(fpl_20))
# %%


# %%
%%time
X = fpl.select_dtypes(include='number').drop('total_points', axis = 1).to_numpy()
y = fpl['total_points'].to_numpy()
gmm = inf.get_gmm(X, y)
select = inf.SelectVars(gmm, selection_mode = 'backward')
# %%
select.fit(X, y, verbose=True)    
select.get_info()
select.plot_mi()
select.plot_delta()
# %%
X_new = select.transform(X, rd=2)
X_new.shape
# %%