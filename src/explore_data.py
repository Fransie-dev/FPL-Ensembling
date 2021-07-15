# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import pandas as pd
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//cleaned_imp.csv', index_col=0)
fig = plt.figure(figsize = (20,20))
ax = fig.gca()
df.select_dtypes(['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).hist(ax=ax)
# %%
#Plot Data
fig = plt.figure(figsize = (20,20))
ax = fig.gca()
sns.distplot(df.select_dtypes(['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
             , bins=25, 
             color="g", ax=ax)
plt.show()


# %%
df.select_dtypes(['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
# %%
df[['GW', 'total_points', 'team_h_difficulty', 'team_h_score',
       'team_h_strength_attack', 'team_h_strength_defense',
       'team_h_strength_overall', 'team_a_difficulty', 'team_a_score',
       'team_a_strength_attack', 'team_a_strength_defense',
       'team_a_strength_overall', 'goals_scored', 'goals_conceded', 'assists',
       'clean_sheets', 'penalties_saved', 'penalties_missed', 'saves',
       'own_goals', 'red_cards', 'yellow_cards', 'minutes', 'selected',
       'value', 'bonus', 'bps', 'transfers_balance', 'transfers_in',
       'transfers_out', 'influence', 'creativity', 'threat', 'ict_index',
       'shots', 'key_passes', 'xG', 'xA', 'npg', 'npxG', 'xGChain',
       'xGBuildup']].drop('GW', 'team_h_dif')
