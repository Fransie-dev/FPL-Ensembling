# %%
import pandas as pd
from pandas.core.frame import DataFrame
import statsmodels.api as sm
from utilities import dlt_create_dir
from clean_data import delete_any_duplicates
import stepwiseSelection as ss
from collections import Counter


def read_data(season):
    """[This function reads the data # Note: Train on 2019-20]

    Args:
        season ([type]): [description]

    Returns:
        [type]: [description]
    """    
    training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
    fpl = pd.read_csv(training_path + 'cleaned_fpl.csv', index_col=0)
    understat = pd.read_csv(training_path + 'cleaned_understat.csv', index_col=0)
    fpl = delete_any_duplicates(fpl)
    understat = delete_any_duplicates(understat)
    return fpl, understat

def backward_elimination(df, threshold_in = 0.05, verbose = True):
    """[This function recursively eliminates features and returns all features that are above the level of significance]

    Args:
        df ([type]): [description]
        threshold_in ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """    
    df = delete_any_duplicates(df)
    X = df.drop(columns= ['total_points', 'player_name', 'kickoff_time', 'GW'])
    y = df.total_points
    cols = list(X.columns)
    while (len(cols) > 0):
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y, X_1.astype(float)).fit()
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
    # return df[['player_name', 'GW', 'kickoff_time','total_points'] + selected_features]
    return selected_features


def forward_elimination(df, threshold_in = 0.05, verbose=True):
    """[This function recursively adds features and returns all features that are above the level of significance]

    Args:
        df ([type]): [description]
        threshold_in ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """    
    df = delete_any_duplicates(df)
    X = df.drop(columns= ['total_points', 'player_name', 'kickoff_time', 'GW'])
    y = df.total_points
    initial_list = []
    selected_features = list(initial_list)
    while True:
        changed=False
        excluded_features = list(set(X.columns)-set(selected_features))
        new_pval = pd.Series(index=excluded_features, dtype='float64')
        for new_column in excluded_features:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[selected_features+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            selected_features.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        if not changed:
            break
    print(f'{len(selected_features)} features selected')
    print(selected_features)
    # return df[['player_name', 'GW', 'kickoff_time','total_points'] + selected_features]
    return selected_features

def stepwise_select(data_str, type, elim_crit):
    if data_str == 'fpl':
        df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//cleaned_fpl.csv', index_col=0)
    if data_str == 'understat':
        df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//cleaned_understat.csv', index_col=0)
    df = delete_any_duplicates(df)
    X = df.drop(columns= ['total_points', 'player_name', 'kickoff_time', 'GW'])
    y = df.total_points
    if type == 'backward':
        final_vars, iterations_logs = ss.backwardSelection(X,y,
                                                           varchar_process='drop', 
                                                           elimination_criteria=elim_crit)
        iterations_file = open(f'C://Users//jd-vz//Desktop//Code//misc//{data_str}_backward_select_logs.txt',"w+") 
        iterations_file.write(iterations_logs)
        iterations_file.close()
    if type == 'forward':
        final_vars, iterations_logs = ss.forwardSelection(X,y,
                                                          varchar_process='drop', 
                                                          elimination_criteria=elim_crit)
        iterations_file = open(f'C://Users//jd-vz//Desktop//Code//misc//{data_str}_forward_select_logs.txt',"w+") 
        iterations_file.write(iterations_logs)
        iterations_file.close()
    return final_vars, iterations_logs
        
def collect_features(be, fe, be_r2, fe_r2, fe_AIC, be_AIC):
    lst = be + fe + be_r2 + fe_r2 + fe_AIC + be_AIC
    selected = pd.DataFrame(Counter(lst).most_common())
    selected = selected[selected[0] != 'intercept']
    return selected

    
def feature_selection(fpl, understat):
    fpl_be = backward_elimination(fpl) 
    understat_be = backward_elimination(understat)
    fpl_fe = forward_elimination(fpl) 
    understat_fe = forward_elimination(understat)
    fpl_fe_r2, iterations_logs = stepwise_select('fpl', 'forward', 'r2')
    understat_fe_r2, iterations_logs = stepwise_select('understat', 'forward', 'r2')
    fpl_be_r2, iterations_logs = stepwise_select('fpl', 'backward', 'r2')
    understat_be_r2, iterations_logs = stepwise_select('understat', 'backward', 'r2')
    fpl_fe_AIC, iterations_logs = stepwise_select('fpl', 'forward', 'AIC')
    understat_fe_AIC, iterations_logs = stepwise_select('understat', 'forward', 'AIC')
    fpl_be_AIC, iterations_logs = stepwise_select('fpl', 'backward', 'AIC')
    understat_be_AIC, iterations_logs = stepwise_select('understat', 'backward', 'AIC')
    fpl_list = fpl_be + fpl_fe  + fpl_fe_r2 + fpl_be_r2 + fpl_fe_AIC + fpl_be_AIC
    understat_list = understat_be + understat_fe  + understat_fe_r2 + understat_be_r2 + understat_fe_AIC + understat_be_AIC
    selected_fpl = pd.DataFrame(Counter(fpl_list).most_common())
    selected_understat = pd.DataFrame(Counter(understat_list).most_common())
    selected_fpl[selected_fpl[0] != 'intercept']
    selected_understat[selected_understat[0] != 'intercept']
    return selected_fpl, selected_understat
    
fpl, understat = read_data(season = '2019-20') # Season stays constant for feature selection
selected_fpl, selected_understat = feature_selection(fpl, understat) # Todo
# %%


# %%
def main():
    # TODO: Add averages and lag data to the match
    
    fpl, understat = read_data(season = '2019-20') # Season stays constant for feature selection
    selected_fpl, selected_understat = feature_selection(fpl, understat) # Todo
    
    # ols_dir = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//OLS//'
    # dlt_create_dir(path=ols_dir)
    # selected_fpl.to_csv(ols_dir + 'selected_fpl_features.csv')
    # selected_understat.to_csv(ols_dir + 'selected_understat_features.csv')
    
    
if __name__ == '__main__':
    main()
    print('Success')
    
    