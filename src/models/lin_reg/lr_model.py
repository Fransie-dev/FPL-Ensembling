# %%
import pandas as pd
from sklearn.linear_model import LinearRegression
from lr_preprocess import preprocess_data
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
def metrics(model, y_test, y_pred):
    """[This function compares the real and predicted data to each other]

    Args:
        y_test ([type]): [The actual, true data]
        y_pred ([type]): [The predicted data]
    """
    print(f'\n{model}: R-squared score = ', r2_score(y_test, y_pred))
    print(f'{model}: Root mean squared error = ', np.sqrt(
        mean_squared_error(y_test, y_pred)), end='\n\n')

def test_LR_model(df, df_test):
    """[This function tests a linear regression model on the provided data]

    Args:
        X_train ([type]): [The training predictors]
        X_test ([type]): [The testing predictors]
        y_train ([type]): [The testing predictors]
        y_test ([type]): [The training response]
    """
    X_train = df.drop(columns = ['total_points'])
    y_train = df['total_points']
    X_test = df_test.drop(columns = ['total_points'])
    y_test = df_test['total_points']
    regressor = LinearRegression(normalize=True,n_jobs=-1)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    metrics('Linear Regression', y_test, y_pred)
    return y_pred

# df = pd.read_csv('C://Users//jd-vz//Desktop//Code//src//models//lin_reg//data//fpl_2019-20_training_data.csv', index_col=0)
df = preprocess_data('fpl', '2019-20', OHE = True, FEAT = True, OUTLIER = False, SCL = None)
df = df.drop(columns = ['player_name', 'GW'])
df_test = preprocess_data('fpl', '2020-21', OHE = True, FEAT = True, OUTLIER = False, SCL = None)
df_test = df_test[df_test['GW'] < 38]
df_test = df_test.drop(columns = ['player_name', 'GW'])
y_pred = test_LR_model(df, df_test)

# %%

true_value = df_test['total_points']
predicted_value = y_pred
plt.figure(figsize=(10,10))
plt.scatter(true_value, predicted_value, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(predicted_value), max(true_value))
p2 = min(min(predicted_value), min(true_value))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()
# %%
