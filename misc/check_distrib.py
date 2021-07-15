# %%
import pandas as pd
import seaborn as sns
from fitter import Fitter, get_common_distributions
# %%
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//training//cleaned_fpl.csv', index_col=0)
df_us = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//training//cleaned_understat.csv', index_col=0)
# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Plot histogram
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# sns.set_style('white')
# sns.set_context("paper", font_scale = 2)
# sns.displot(data=df, x="total_points", kind="hist", bins = 100, aspect = 1.5)
# %%
def plot_feat(df, feat):
    sns.set_style('white')
    sns.set_context("paper", font_scale = 2)
    sns.displot(data=df, x=feat, kind="hist", bins = 100, aspect = 1.5)
# %%
# plot_feat(df, 'ict_index')
# %%


# %%

# %%
def check_data_dist(df, feat = None):
    if feat is None:
        for feat in df.select_dtypes(['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns:
            dist_fit = Fitter(df[feat], timeout=60*10, distributions=get_common_distributions())
            dist_fit.fit()
            # dist_fit.summary() 
            key, value = list(dist_fit.get_best(method = 'sumsquare_error').items())[0] # Key is the identified distribution
            print(f'{feat} has {key} distribution')
    else:
            dist_fit = Fitter(df[feat], timeout=60*10, distributions=get_common_distributions())
            dist_fit.fit()
            dist_fit.summary() # Plots distribution 
            key, value = list(dist_fit.get_best(method = 'sumsquare_error').items())[0] # Key is the identified distribution
            print(f'{feat} has {key} distribution')
            
check_data_dist(df_us, 'xG')
# %%
dist_fit = Fitter(df['total_points'], timeout=60*10, distributions=get_common_distributions())
dist_fit.fit()
dist_fit.summary()

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


calculate_vif(df.select_dtypes(['int16', 'int32', 'int64', 'float16', 'float32', 'float64']))
# %%
