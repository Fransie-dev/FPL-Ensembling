# %%
sys.path.insert(0, 'C://Users//jd-vz//Desktop//Code//src//')
import pandas as pd
import statsmodels.api as sm
from collections import Counter
import sys
import stepwiseSelection as ss
from utilities import dlt_create_dir

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
    understat_imp = pd.read_csv(training_path + 'cleaned_imp.csv', index_col=0)
    return fpl, understat, understat_imp


def backward_elimination(df, data_str, threshold_in = 0.05, verbose = True):
    """[This function recursively eliminates features and returns all features that are above the level of significance]

    Args:
        df ([type]): [description]
        threshold_in ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """    
    data_str = 'fpl'
    elim_crit = 't_stat'
    iterations_log = ""
    X = df.drop(columns= ['total_points', 'player_name', 'kickoff_time', 'GW'])
    y = df.total_points
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
    # return df[['player_name', 'GW', 'kickoff_time','total_points'] + selected_features]
    return selected_features

def write_log(iterations_log, data_str, elim_crit, direction): #TODO: incorporate me
    log_dir = 'C://Users//jd-vz//Desktop//Code//src//models//lin_reg//misc//'
    iterations_file = open(log_dir + f'{data_str}_{elim_crit}_{direction}_select_logs.txt',"w+") 
    iterations_file.write(iterations_log)
    iterations_file.close()


def forward_elimination(df, data_str, threshold_in = 0.05, verbose=True):
    """[This function recursively adds features and returns all features that are above the level of significance]

    Args:
        df ([type]): [description]
        threshold_in ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """    
    
    X = df.drop(columns= ['total_points', 'player_name', 'kickoff_time', 'GW'])
    y = df.total_points
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
    # return df[['player_name', 'GW', 'kickoff_time','total_points'] + selected_features]
    return selected_features

def stepwise_select(df, data_str, type, elim_crit):
    """[This model utilizes Apache's forward and backward selection based on three different criterion]


    Args:
        data_str ([type]): [description]
        type ([type]): [description]
        elim_crit ([type]): [description]  ["adjr2", "aic", "bic", "r2"]

    Returns:
        [type]: [description]
    """    

    X = df.drop(columns= ['total_points', 'player_name', 'kickoff_time', 'GW'])
    Y = df.total_points
    if type == 'backward':
        final_vars, iterations_logs = ss.backwardSelection(X,Y, elimination_criteria=elim_crit)
        write_log(iterations_logs, data_str, elim_crit, direction = 'backward')
    if type == 'forward':
        final_vars, iterations_logs = ss.forwardSelection(X,Y, elimination_criteria=elim_crit)
        write_log(iterations_logs, data_str, elim_crit, direction = 'forward')
    return final_vars
        
def collect_features(be_t, fe_t, be_r2, fe_r2, be_adjr2, fe_adjr2, fe_AIC, be_AIC ,fe_BIC, be_BIC):
    lst = be_t +  fe_t +  be_r2 +  fe_r2 +  be_adjr2 +  fe_adjr2 +  fe_AIC +  be_AIC  + fe_BIC +  be_BIC
    selected = pd.DataFrame(Counter(lst).most_common())
    selected = selected[selected[0] != 'intercept']
    selected.rename(columns = {0:'Feature', 1:'Occurences'}, inplace = True)
    return selected


def one_hot_encode(fpl):
    """[This function one hot encodes the four categorical features into dummy variables]

    Args:
        fpl ([type]): [description]

    Returns:
        [type]: [description]
    """    
    fpl = pd.get_dummies(fpl, columns=['was_home', 'position', 'team_h', 'team_a'], 
                      prefix=['home', 'position', 'team_h', 'team_a'])
    fpl.drop(columns=['home_False'], axis=1, inplace=True)
    return fpl

fpl, understat, understat_imp = read_data(season = '2019-20') # Season stays constant for feature selection

def feature_selection(df, data_str, required_votes):
    data = one_hot_encode(df)
    log_dir = 'C://Users//jd-vz//Desktop//Code//src//models//lin_reg//misc//'
    be_t = backward_elimination(data, data_str)
    fe_t = forward_elimination(data, data_str) 
    be_r2 = stepwise_select(data, data_str, 'backward', 'r2') 
    fe_r2 = stepwise_select(data, data_str, 'forward', 'r2') 
    be_adjr2 = stepwise_select(data, data_str, 'backward', 'adjr2') 
    fe_adjr2 = stepwise_select(data, data_str, 'forward', 'adjr2') 
    be_AIC = stepwise_select(data, data_str, 'backward', 'AIC') 
    fe_AIC = stepwise_select(data, data_str, 'forward', 'AIC') 
    be_BIC = stepwise_select(data, data_str, 'backward', 'BIC') 
    fe_BIC = stepwise_select(data, data_str, 'forward', 'BIC') 
    selected = collect_features(be_t, fe_t, be_r2, fe_r2, be_adjr2, fe_adjr2, fe_AIC, be_AIC ,fe_BIC, be_BIC)
    selected.to_csv(log_dir + f'{data_str}_feature_pool.csv') # updt:07/3
    selected.Feature[selected.Occurences >= required_votes].to_csv(log_dir + f'selected_{data_str}_features.csv')
    
def main(required_votes):
    # TODO: Add averages and lag data to the match
    fpl, understat, imp = read_data(season = '2019-20') # Season stays constant for feature selection
    feature_selection(fpl, 'fpl', required_votes)
    feature_selection(understat, 'understat', required_votes)
    feature_selection(imp, 'imp', required_votes)


    
if __name__ == '__main__':
    main(required_votes = 3) # 3/6 of models need to vote for a feature)
    print('Success')
    
# %%


# %%

    