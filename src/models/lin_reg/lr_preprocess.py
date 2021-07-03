# %%
import sys
from pandas.core.frame import DataFrame
sys.path.insert(0, 'C://Users//jd-vz//Desktop//Code//src//')
import pandas as pd
from utilities import one_hot_encode
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from utilities import dlt_create_dir
from merge_data import intersect, union

def scale_numeric(df):
    total_points = df['total_points']
    std_scaler = StandardScaler()
    df_scale = df.drop(columns = ['total_points'], axis = 1)
    df_scaled = std_scaler.fit_transform(df_scale.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=df_scale.columns) 
    df_scaled['total_points'] = total_points
    return df_scaled
    
def outliers_removal(df, type):
    # NB: This function removes too many outlying points
    numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.select_dtypes(include=numeric_types).columns:
        if type == 'IQR':
            Q1= df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q3 + 1.5 * IQR
            lower_limit = Q1 - 1.5 * IQR
        if type == 'SD':
            upper_limit = df[col].mean() + 3 * df[col].std()
            lower_limit = df[col].mean() - 3 * df[col].std()
        df_len = len(df)
        df.drop(df[(df[col] > upper_limit) | (df[col] < lower_limit)].index , inplace=True)
        print(f'Removed {df_len - len(df)} entries from {col} with the {type} method')
    return df
   
def feat_removal(df, lin_reg_dir, data_str):
    feats = pd.read_csv(lin_reg_dir + f'misc//selected_{data_str}_features.csv', index_col=0)
    df_21 = one_hot_encode(pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//training//' + f'cleaned_{data_str}.csv', index_col=0))
    cut = intersect(df.columns, df_21.columns)
    df = df[cut]
    print(df.columns)
    return df, feats

def valid_features(data_str, lin_reg_dir):
    df_20 = one_hot_encode(pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//' + f'cleaned_{data_str}.csv', index_col=0)).columns
    df_21 = one_hot_encode(pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//training//' + f'cleaned_{data_str}.csv', index_col=0)).columns
    feats = pd.read_csv(lin_reg_dir + f'misc//selected_{data_str}_features.csv', index_col=0)
    feat_pool = intersect(df_20, df_21)
    valid_feats = intersect(feats.Feature, feat_pool)
    valid_feats = valid_feats + ['total_points']
    return valid_feats

def preprocess_data(data_str, season, OHE = True, FEAT = True,OUTLIER = None, SCL = None):
    #! Confirm the order of this program with Thorstens 
    #! Linear Regression: One hot encode categorical --> Apply Feature Selection Results --> Remove Outliers --> Scale all features
    train_dir = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//' 
    lin_reg_dir = 'C://Users//jd-vz//Desktop//Code//src//models//lin_reg//'
    df = pd.read_csv(train_dir + f'cleaned_{data_str}.csv', index_col=0)
    feats = pd.read_csv(lin_reg_dir + f'misc//selected_{data_str}_features.csv', index_col=0)
    if OHE == True:
        df = one_hot_encode(df)
    if FEAT == True:
       val_feat = valid_features('fpl', lin_reg_dir)
       df = df[val_feat + ['player_name', 'GW']]
    if OUTLIER == True:
        df[val_feat] = outliers_removal(df[val_feat], type = 'SD')
    if SCL == True:
        df[val_feat] = scale_numeric(df[val_feat])
    # df.to_csv(lin_reg_dir + 'data//' f'{data_str}_{season}_training_data.csv')
    return df

def main(season):
    for data_str in ['fpl', 'understat', 'imp']:
        preprocess_data(data_str, season, OHE = True, FEAT = True, OUTLIER = False, SCL = True) 

if __name__ == '__main__':
    dlt_create_dir('C://Users//jd-vz//Desktop//Code//src//models//lin_reg//data')
    main('2019-20')
    main('2020-21')
# %%
    
    
    
DataFrame.shape