# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Read and Scale Numerics
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//cleaned_imp.csv', index_col=0)
scaler = StandardScaler()
df[df.drop(columns=['GW']).select_dtypes("number").columns] = scaler.fit_transform(df.drop(columns=['GW']).select_dtypes("number"))
# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# OHE categorical
df = pd.get_dummies(df, columns=['was_home', 'position', 'team_h', 'team_a'], 
                    prefix=['home', 'position', 'team_h', 'team_a'])
df.drop(columns=['home_False'], axis=1, inplace=True)
# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Split
X = df.drop(columns=['GW', 'player_name', 'total_points', 'kickoff_time'])
y = df["total_points"]
# %%
%%time
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>#
# Try RFE
random_state = 0 
# estimator = LinearRegression()
estimator = SVR(C=100, epsilon=0.01, gamma=0.005, verbose=True, kernel='rbf') # Note: This was fitted to scaled data
rfecv = RFECV(estimator=estimator, cv=3,verbose = 2)
rfecv.fit(X, y)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
plt.grid()
plt.xticks(range(1, X.shape[1]+1))
plt.xlabel("Number of Selected Features")
plt.ylabel("CV Score")
plt.title("Recursive Feature Elimination (RFE)")
plt.show()
print("The optimal number of features: {}".format(rfecv.n_features_))

# %%
X_rfe = X.iloc[:, rfecv.support_] # Selected features
# %%
print("\"X\" dimension: {}".format(X.shape))
print("\"X\" column list:", X.columns.tolist())
print("\"X_rfe\" dimension: {}".format(X_rfe.shape))
print("\"X_rfe\" column list:", X_rfe.columns.tolist())
# %%
import thundersvm
# %%
from thundersvm import SVR
# %%
