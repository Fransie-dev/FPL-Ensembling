# https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/import sys
# %%
from fitter.fitter import get_distributions
from fitter import Fitter, get_common_distributions, get_distributions
import seaborn as sns
import pandas as pd
import numpy as np

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))


def check_data_dist(df, feat = None, bins = 100, distrib = ['expon', 'norm', 'uniform']):
    if feat is None:
        cf, dist, sse = [], [], []
        for feat in df.select_dtypes('number').columns:
            dist_fit = Fitter(df[feat], timeout=60*10, distributions=distrib, bins=bins)
            dist_fit.fit()
            key, value = list(dist_fit.get_best(method = 'sumsquare_error').items())[0] # Key is the identified distribution
            cf.append(feat), dist.append(key), sse.append(value)
            # print(f'{feat} has {key} distribution with {value} SSE')
        df_fitted = pd.DataFrame({'Feature': cf, 'Distribution': dist })
        return df_fitted
    else:
            # sns.set_style('white')
            # sns.set_context("paper", font_scale = 2)
            # sns.displot(data=df, x=feat, kind="hist", bins = bins, aspect = 1.5)
            dist_fit = Fitter(df[feat], bins = bins, timeout=60*10, distributions=distrib)
            dist_fit.fit()
            print((dist_fit.summary(Nbest=3)).iloc[:,0].to_latex()) # Plots distribution 

df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_fpl.csv')
distrib_fit = check_data_dist(df)
df_2 = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_us.csv')
# distrib_fit_2 = check_data_dist(df_2)
# distrib_fit_2['FPL']  = distrib_fit['Distribution']
# print(distrib_fit_2.to_latex())



# %%

from fitter import HistFit
hf = HistFit(df['value'].to_list(), bins=100)
hf.fit(error_rate=0.03, Nfit=20)
print(hf.mu, hf.sigma, hf.amplitude)
# # %%

# # %%
# get_common_distributions()
# # %%

# %%
from keras.wrappers.scikit_learn import KerasRegressor
