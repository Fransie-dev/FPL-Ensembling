# %%
from os import name
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# %%

def metrics(model, y_test, y_pred):
    """[This function compares the real and predicted data to each other]

    Args:
        y_test ([type]): [The actual, true data]
        y_pred ([type]): [The predicted data]
    """
    print(f'\n{model}: R-squared score = ', r2_score(y_test, y_pred))
    print(f'{model}: Mean squared error = ',
          mean_squared_error(y_test, y_pred))
    print(f'{model}: Root mean squared error = ', np.sqrt(
        mean_squared_error(y_test, y_pred)), end='\n\n')

def test_LR_model(X_train, X_test, y_train, y_test):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    regressor = LinearRegression(normalize=True,n_jobs=-1)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    metrics('Linear Regression', y_test, y_pred)
    return y_pred_train



def test_tree_model(X_train, X_test, y_train, y_test):
    """[This function tests a random forest model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    tree = RandomForestRegressor(n_estimators=200)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    metrics('Random Forest Regressor', y_test, y_pred)
    
    
    
season = '2019-20'
training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
fpl = pd.read_csv(training_path + 'cleaned_fpl.csv', index_col=0)
understat = pd.read_csv(training_path + 'cleaned_understat.csv', index_col=0)
fpl_feat = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//OLS/selected_fpl_features.csv', index_col=0)
understat_feat = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//OLS/selected_understat_features.csv', index_col=0)
df_fpl = fpl[fpl_feat['Feature'].to_list() + ['total_points']].copy()


# %%
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)                   
# lr = LinearRegression(normalize=False,n_jobs=-1)
# lr.fit(X_train, y_train)
# y_pred = lr.predict(X_test)
# # y_pred_train = lr.predict(X_train)
# metrics('Linear Regression', y_test, y_pred)

# # %%
# feat = 'ict_index'
# plt.scatter(X_train[feat], y_train, color = "red")
# plt.scatter(X_train[feat], lr.predict(X_train), color = "green")
# plt.title(f"Total points versus {feat}")
# plt.xlabel(f"{feat}")
# plt.ylabel("Total Points")
# plt.show()
# %%
X = df_fpl.drop(columns=['total_points'])                   
Y = df_fpl['total_points']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)                   
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
# %%
























# %%
feat = 'ict_index'
plt.scatter(x=X_train[feat], y=y_train)
# %%
plt.xlabel(feat)
plt.ylabel(y_train.name)
plt.plot(X_train[feat], y_pred_train, color = 'red')
plt.show()
# %%

# %%
df_viz = df_fpl.copy()
df_viz['total_points'] = fpl.total_points

sns.lmplot(x="threat", y="total_points", hue = 'home_True', data=fpl)

# %%
import seaborn as sns; sns.set_theme(color_codes=True)
ax = sns.regplot(x="threat", y="total_points", data=df_viz)
# %%
def viz_corr(X, Y):
    corrmat = df_viz.corr() 
    f, ax = plt.subplots(figsize =(20, 20))
    sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) # Multicollinearity
# %%
