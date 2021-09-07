# %%
import pandas as pd
from utilities import one_hot_encode
from sklearn.preprocessing import StandardScaler

def read_data():
    fpl21 = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//training//shifted_fpl.csv').dropna()
    fpl20 = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//shifted_fpl.csv').dropna()
    us21 = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//training//shifted_us.csv').dropna()
    us20 = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//shifted_us.csv').dropna()
    return fpl21, fpl20, us21, us20

def one_hot_encode(df):
    """[This function one hot encodes the four categorical features into dummy variables]

    Args:
        fpl ([type]): [description]

    Returns:
        [type]: [description]
    """    
    print(df.select_dtypes(include='object').columns)
    df = pd.get_dummies(df, columns=['was_home', 'position', 'team_h', 'team_a'], 
                      prefix=['home', 'position', 'team_h', 'team_a'])
    df.drop(columns=['home_False'], axis=1, inplace=True)
    return df

def scale_numeric(df):
    std_scaler_X = StandardScaler()
    std_scaler_Y = StandardScaler()
    df_scaled = pd.DataFrame(std_scaler_X.fit_transform(df.drop(columns = ['total_points'], axis = 1).values), columns=df.drop(columns = ['total_points'], axis = 1).columns, index=df.index)
    df_scaled['total_points'] = std_scaler_Y.fit_transform(df['total_points'].to_numpy().reshape(-1, 1))
    return df_scaled, std_scaler_X, std_scaler_Y




# %%
fpl21 = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//training//cleaned_fpl.csv').dropna()
fpl21.shape
fpl21.columns
# %%
one_hot_encode(fpl21)


# %%






# %%
# fpl21.select_dtypes(include='object')
# fpl21.select_dtypes(include='number')
# %%
