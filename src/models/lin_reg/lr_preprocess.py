# %%
import sys
sys.path.insert(0, 'C://Users//jd-vz//Desktop//Code//src//')
import pandas as pd
from utilities import one_hot_encode
from sklearn.preprocessing import StandardScaler

def intersect(a, b):
    """[This function finds the intersection between two player name columns]

    Args:
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [The intersection]
    """    
    # print(len(list(set(a) & set(b))), 'unique and matching names between FPL and Understat')
    return list(set(a) & set(b))

def scale_numeric(df):
    std_scaler_X = StandardScaler()
    std_scaler_Y = StandardScaler()
    df_scaled = pd.DataFrame(std_scaler_X.fit_transform(df.drop(columns = ['total_points'], axis = 1).values), columns=df.drop(columns = ['total_points'], axis = 1).columns, index=df.index)
    df_scaled['total_points'] = std_scaler_Y.fit_transform(df['total_points'].to_numpy().reshape(-1, 1))
    return df_scaled, std_scaler_X, std_scaler_Y
    
def outliers_removal(df, type):
    # NB: This function removes too many outlying points and should be reconsiderd
    #! Assumes normal distribution
    #* Package that investigates the data distribution... exp, beta, gamma... technomatex
    #* Exp --> upper qnt
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
   
# updt: changed cleaned to shifted 
def valid_features(data_str, lin_reg_dir):
    df_20 = one_hot_encode(pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//' + f'shifted_{data_str}.csv')).columns
    df_21 = one_hot_encode(pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//training//' + f'shifted_{data_str}.csv')).columns
    feats = pd.read_csv(lin_reg_dir + f'misc//selected_{data_str}_features.csv', index_col=0) # Linear Regression selected features
    feat_pool = intersect(df_20, df_21)
    valid_feats = intersect(feats.Feature, feat_pool)
    valid_feats = valid_feats + ['total_points'] # Features within both datasets, and the target
    return valid_feats

def preprocess_data(data_str, season, OHE = True, FEAT = True, OUTLIER = None, SCL = True):
    #! Confirm the order of this program with Thorstens 
    #! Linear Regression: One hot encode categorical --> Apply Feature Selection Results --> Remove Outliers --> Scale all features
    train_dir = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//' 
    lin_reg_dir = 'C://Users//jd-vz//Desktop//Code//src//models//lin_reg//'
    df = pd.read_csv(train_dir + f'shifted_{data_str}.csv')
    if OHE == True:
        df = one_hot_encode(df)
    if FEAT == True:
       val_feat = valid_features('fpl', lin_reg_dir)
       df = df[val_feat + ['player_name', 'GW']]
    if OUTLIER == True:
        df[val_feat] = outliers_removal(df[val_feat], type = 'IQR') # Not recommended
    if SCL == True:
        df[val_feat], std_scaler_X, std_scaler_Y = scale_numeric(df[val_feat])
        return df,  std_scaler_X, std_scaler_Y, val_feat
    return df
# %%