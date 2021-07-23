
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import OrdinalEncoder
random_state = 42
X_train2_sup = X_train2.copy() #deep copy
# %%
%%time
X_model, X_valid, y_model, y_valid = train_test_split(X_train2_sup, y_train, stratify=y_train, random_state=random_state, test_size=.8)

model_dict = {'LogisticRegression': LogisticRegression(penalty='l1', solver='saga', C=2, multi_class='multinomial', n_jobs=-1, random_state=random_state)
             , 'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=200, max_depth=3, min_samples_leaf=.06, n_jobs=-1, random_state=random_state)
              , 'RandomForestClassifier': RandomForestClassifier(n_estimators=20, max_depth=2, min_samples_leaf=.1, random_state=random_state, n_jobs=-1)
             }
estimator_dict = {}
importance_fatures_sorted_all = pd.DataFrame()
for model_name, model in model_dict.items():
    print('='*10, model_name, '='*10)
    model.fit(X_model, y_model)
    print('Accuracy in training:', accuracy_score(model.predict(X_model), y_model))
    print('Accuracy in valid:', accuracy_score(model.predict(X_valid), y_valid))
    importance_values = np.absolute(model.coef_) if model_name == 'LogisticRegression' else model.feature_importances_
    importance_fatures_sorted = pd.DataFrame(importance_values.reshape([-1, len(X_train2_sup.columns)]), columns=X_train2_sup.columns).mean(axis=0).sort_values(ascending=False).to_frame()
    importance_fatures_sorted.rename(columns={0: 'feature_importance'}, inplace=True)
    importance_fatures_sorted['ranking']= importance_fatures_sorted['feature_importance'].rank(ascending=False)
    importance_fatures_sorted['model'] = model_name
    print('Show top 10 important features:')
    display(importance_fatures_sorted.drop('model', axis=1).head(10))
    importance_fatures_sorted_all = importance_fatures_sorted_all.append(importance_fatures_sorted)
    estimator_dict[model_name] = model

plt.title('Feature importance ranked by number of features by model')
sns.lineplot(data=importance_fatures_sorted_all, x='ranking', y='feature_importance', hue='model')
plt.xlabel("Number of features selected")















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
