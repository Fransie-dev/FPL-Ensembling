# %%
# This script serves for the visual EDA
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
data_file =  'C://Users//jd-vz//Desktop//Code//data//collected_us.csv'
# %%
# * 1. The total team points scored in different seasons (Complete)
sns.set(rc={"figure.figsize":(12, 6)}) 
df = pd.read_csv(data_file)
df = df[df['season'] == 2020] # Change season here
df['TeamTotalPointsCount'] = df.groupby(['team', 'season'])['total_points'].transform('sum')
df['TeamTotalCount'] = df.groupby(['team', 'season'])['total_points'].transform('sum')
df['TeamTotalCostCount'] = df.groupby(['team', 'season'])['value'].transform('sum')
df.sort_values('TeamTotalCount',ascending=False, inplace=True);
df = df[['team', 'TeamTotalPointsCount', 'TeamTotalCount','season', 'TeamTotalCostCount']].drop_duplicates()
ax = sns.barplot(x="team", y="TeamTotalPointsCount", data=df, ci=None, label = 'Total team points scored', palette =sns.color_palette('Reds_r', 40))
plt.xticks(rotation=90)
plt.xlabel('Teams')
plt.ylabel('Total team points')
width_scale = 0.5
for bar in ax.containers[0]:
    bar.set_width(bar.get_width() * width_scale)
ax2 = ax.twinx()
ax_3 = sns.barplot(x='team', y='TeamTotalCostCount', label = 'Total team cost', data=df, ci=None,ax=ax2, color = 'gray')
plt.ylabel('Total team cost')
# ax_3.legend(loc='upper right')
for bar in ax2.containers[0]:
    x = bar.get_x()
    w = bar.get_width()
    bar.set_x(x + w * (1- width_scale))
    bar.set_width(w * width_scale)
plt.grid(False)
handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax_3.get_legend_handles_labels())]
ax_3.legend(handles, labels, loc='upper right')
plt.show()
# %%
# * 3. The total points scored by positions in different seasons (Complete)
# * Plus the effect of supporters on player performance
df = pd.read_csv(data_file)
ax_1 = sns.catplot(x="position", y="total_points", hue="was_home", ci=90, col = 'season', col_order = [2019,2020], n_boot=1000, kind="point", data=df)
ax_1.set_xlabels('Position', fontsize=15) # not set_label
ax_1.set_ylabels('Average points scored', fontsize=15) # not set_label
ax_1._legend.texts[0].set_text("Home")
ax_1._legend.texts[0].set_text("Away")
df['Supporters'] = np.where(df['kickoff_time'] < '2020-03-13', True, False)
ax_2 = sns.catplot(x="position", y="total_points", col="Supporters", col_order = [True, False], hue="was_home", ci=90, n_boot=1000, kind="point", data=df)
ax_2.set_xlabels('Position', fontsize=15) # not set_label
ax_2.set_ylabels('Average points scored', fontsize=15) # not set_label
ax_2._legend.texts[0].set_text("Home")
ax_2._legend.texts[0].set_text("Away")
# %%
# * 4a. Unique team counts (Complete)
sns.set_style('darkgrid')
pal = sns.dark_palette("#69d", reverse=True, as_cmap=False,n_colors=40)
df = pd.read_csv(data_file)
ax = sns.countplot(x='team',  data=df, palette= pal, order=df['team'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Teams')
plt.ylabel('Number of matches')
plt.show()
# %%
# * 4b. Number of unique players in each team (Complete)
df = pd.read_csv(data_file)
cnts = df.groupby(['team', 'player_name']).size().reset_index(name='counts')
ax = sns.countplot(x='team',  data=cnts, palette=pal, order=cnts['team'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Teams')
plt.ylabel('Number of unique players')
plt.show()
# %%
# * 4c) The number of times unique players played in gameweeks (Complete)
# * Plus the reason why some are so high
df = pd.read_csv(data_file)
df = df.groupby(['team', 'player_name', 'GW', 'season']).size().reset_index(name='counts') 
df['total_player_counts'] = df.groupby(['team', 'season', 'GW'])['counts'].transform('sum')
df.sort_values(['total_player_counts'], ascending=False, inplace=True)
sns.catplot(x = 'team', y = 'total_player_counts', hue = 'season', kind = 'boxen', data = df, aspect =4, height = 6)
plt.xlabel('Teams')
plt.xticks(rotation=90)
plt.ylabel('Distribution of number of unique players used in gameweeks')
df = pd.read_csv(data_file)
df = df.groupby(['team', 'opponent_team', 'kickoff_time','player_name', 'GW', 'season', 'was_home', 'team_h_score', 'team_a_score']).size().reset_index(name='counts') 
df['total_player_counts'] = df.groupby(['team', 'season', 'GW'])['counts'].transform('sum')
df = df.loc[df.season == 2020,:]
df = df.loc[df.team == 'Man Utd',:]
df = df.loc[df.GW == 35,:]
df[['team', 'opponent_team','kickoff_time', 'was_home', 'team_h_score', 'team_a_score']].drop_duplicates()
print(len(df['player_name'].unique()), 'unique players for MUtd in GW 35')
# %%
# * 4c) The number of times positions played in gameweeks (Complete)
# * Plus only one forward is required
df = pd.read_csv(data_file)
df = df.groupby(['position', 'team', 'kickoff_time']).size().reset_index(name='counts') 
df['total_position_counts'] = df.groupby(['position', 'team', 'kickoff_time'])['counts'].transform('sum')
df.sort_values(['total_position_counts'], ascending=False, inplace=True)
sns.catplot(x = 'position', y = 'total_position_counts', hue = 'season', kind = 'boxen', data = df, aspect =3, height = 6)
plt.xlabel('Position')
plt.xticks(rotation=90)
plt.ylabel('Distribution of positions used in gameweeks')
# %%
# * 4e) The number of unique fixtures for all teams (Complete)
df = pd.read_csv(data_file)
seas = 2019
df = df.loc[df['season'] == seas,:]
print(df.loc[df.season == seas,['team', 'opponent_team','kickoff_time']].drop_duplicates().shape[0], 'fixtures') # 760 fixtures and each team plays 38 matches
print(df.loc[df.season == seas,['team', 'opponent_team','kickoff_time']].drop_duplicates().shape[0]/len(df['team'].unique()), 'matches by each team') # 760 fixtures and each team plays 38 matches
# * Plus the number of unique fixtures each gameweek   (Complete)
df = pd.read_csv(data_file)
df = df[['team', 'opponent_team','kickoff_time', 'GW', 'season']].drop_duplicates()
df = df.groupby(['GW', 'season']).size().reset_index(name='Num_Matches_GW') 
ax = sns.catplot(x="GW", y="Num_Matches_GW", hue='season', kind = 'bar', aspect = 2, height = 6, data=df, legend = True)
plt.xlabel('Gameweek')
plt.ylabel('The total number of unique fixtures each gameweek')
ax._legend.set_title('Season')
plt.show()
# * Plus a stacked barplot showing the number of fixtures played each gameweek by teams  (Complete)
df = pd.read_csv(data_file)
df = df[['team', 'opponent_team','kickoff_time', 'GW', 'season']].drop_duplicates()
df = df.groupby(['GW', 'team', 'season']).size().reset_index(name='Num_Matches_GW') # No consideration of season with this way
ax = sns.histplot(df, x='GW', hue='team',weights='Num_Matches_GW', multiple='stack',bins=38, palette='tab20c',legend=False)
plt.xlabel('Gameweek')
plt.ylabel('The number of fixtures played')
plt.show()
# %%
# *4f) Clustered heatmap of features (Complete)
# Plus can remove
df = pd.read_csv(data_file)
sns.set_theme()
sns.clustermap(df.select_dtypes(include ='number').corr(), figsize=(20,20))
corr_train = df.select_dtypes(include = 'number').corr().abs().sort_values('total_points', ascending = False)
print(pd.DataFrame(corr_train['total_points'].head(15)))
df = pd.read_csv(data_file)
df = df.drop('total_points', axis=1)
corr_train = df.select_dtypes(include = 'number').corr()
# Set the threshold
threshold = 0.85
# Empty dictionary to hold correlated variables
above_threshold_vars = {}
# For each column, record the variables that are above the threshold
for col in corr_train:
    above_threshold_vars[col] = list(corr_train.index[corr_train[col] > threshold])
# Track columns to remove and columns already examined
cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []

# Iterate through columns and correlated columns
for key, value in above_threshold_vars.items():
    # Keep track of columns already examined
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            # Only want to remove one in a pair
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)
            
cols_to_remove = list(set(cols_to_remove))
print('Number of columns to remove: ', len(cols_to_remove))
print('Columns that are collinear with other columns:', cols_to_remove)

# %%
# *6) Distribution of kickoff times (Complete)
# # Suspended 13 March 2020 to 17 June
import matplotlib.dates as mdates
plt.figure(figsize=(20, 10))
df = pd.read_csv(data_file)
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
ax = sns.kdeplot(x = 'kickoff_time', data = df,hue='season',shade = True,cut = 0.8,clip_on=False,cumulative = True, fill=True, alpha=0.5,  palette =sns.color_palette('rocket', 2)) 
fmt_half_year = mdates.WeekdayLocator(interval=2)
ax.xaxis.set_major_locator(fmt_half_year)
fmt_month = mdates.WeekdayLocator(interval=2)
ax.xaxis.set_minor_locator(fmt_month)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.grid(True)
plt.xticks(rotation=45)
plt.xlabel('Kickoff Time')
plt.ylabel('Density')
plt.show()
# print(min(df.loc[df['season'] == 2020, 'kickoff_time']))
# print(max(df.loc[df['season'] == 2019, 'kickoff_time']))
# %%
# *6a) Plot ploints scored in years(Complete)
df = pd.read_csv(data_file)
plt.figure(figsize=(20, 10))
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
df = df.set_index('kickoff_time')
df['Year'] = df.index.year
sns.boxplot(data=df, x = 'Year', y='total_points')
# *6b) Plot ploints scored in months
plt.figure(figsize=(20, 10))
df['Month'] = df.index.month
sns.boxplot(data=df, x = 'Month', y='total_points')
# *6c) Plot ploints scored in days
plt.figure(figsize=(20, 10))
df['Weekday Name'] = df.index.day_name()
sns.boxplot(data=df, x = 'Weekday Name', y='total_points')
# %%
# *7 Plot the top and bottom players according to total points scored, average minutes, average cost and average bps
sns.set_style('darkgrid')
df = pd.read_csv(data_file)
pal = sns.dark_palette("#69d", reverse=True, as_cmap=False,n_colors=100)
df['player_season_points'] = df.groupby(['player_name', 'season'])['total_points'].transform('sum') # All player points in a season
df['player_season_cost'] = df.groupby(['player_name', 'season'])['value'].transform('mean') # Mean cost throughout the season
df['player_season_minutes'] = df.groupby(['player_name', 'season'])['minutes'].transform('mean') # All minutes played throughout the season
df['player_season_bps'] = df.groupby(['player_name', 'season'])['bps'].transform('mean') # Mean cost throughout the season
df = df[['player_name','season', 'player_season_points', 'player_season_bps', 'player_season_cost',  'player_season_minutes']].drop_duplicates()
df = pd.concat([df.sort_values(['player_season_points'],ascending=False).head(50), df.sort_values(['player_season_minutes'],ascending=False).tail(10)])
plt.figure(figsize=(15, 8))
ax = sns.barplot(x='player_name', y='player_season_points', palette = pal,  label = 'Total points scored in season', data = df, ci=None)
plt.xticks(rotation=90)
plt.xlabel('Player name')
plt.ylabel('Total points scored in season')
plt.grid(False)
ax2 = ax.twinx()
sns.lineplot(y = df['player_season_cost'], x = df['player_name'], marker = 'o',sort = True, ax=ax2,color = '#900c2c', label = 'Average cost in season',err_style="bars", ci=90)
sns.lineplot(y = df['player_season_minutes'],x = df['player_name'], marker='o', sort = True, ax=ax2, color = '#171717', label = 'Average minutes in season',err_style="bars", ci=90)
sns.lineplot(y = df['player_season_bps'], x = df['player_name'],marker='o', sort = True, ax=ax2, color = '#cce7f1',  label = 'Average bonus points scored in season',err_style="bars", ci=90)
plt.xticks(rotation=90)
plt.ylabel('Average value in season')
handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
ax2.legend(handles, labels, loc='upper right')
plt.grid(False)
plt.show()
# %%
# * 8 Plot Kevins value as the gameweeks increase
import seaborn as sns
sns.set_style('darkgrid')
plt.figure(figsize=(15, 8))
pal = sns.dark_palette("#69d", reverse=True, as_cmap=False,n_colors=150)
df = pd.read_csv(data_file)
df = df.loc[(df['player_name'] == 'Kevin De Bruyne') & (df['season'] == 2019)] # The season in which Kevin performed so well 
ax = sns.barplot(x='GW', y='total_points', data = df, palette = pal,ci=None)
ax2 = ax.twinx()
sns.pointplot(y = df['value'], x = df['GW'], marker = 'o',sort = True, ax=ax2,color = '#171717', label = 'Average cost in 2020', ci = None)
plt.grid(False)
handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
ax2.legend(handles, labels, loc='upper right')
plt.show()
# %%
# * 10 Plot position occurrences against home
sns.set(style='darkgrid')
df = pd.read_csv(data_file)
plt.figure(figsize=(10,5))
ax = sns.countplot(x='position', data=df, hue = 'was_home')
# %%
# * 10 Plot some position statistics
df = pd.read_csv(data_file)
df = df[['position', 'influence',  'bps', 'threat', 'creativity', 'ict_index', 'total_points', 'goals_conceded']]
df.groupby('position').sum().T.plot(kind='bar', stacked=True)
# %%
# * 10 Plot some position understat statistics
sns.set(style='darkgrid')
plt.figure(figsize=(10,5))
df = pd.read_csv(data_file)
df = df[['position', 'xGChain', 'xGBuildup', 'xG', 'npxG', 'npg',   'xA']]
df.groupby('position').sum().T.plot(kind='bar', stacked=True)
# %%
# * All plots against position and home
df = pd.read_csv(data_file)
for col in df.select_dtypes(include = 'number'):
    g = sns.FacetGrid(df, col="position", height=5, aspect=1, row = 'was_home')
    g.map(sns.histplot, col)
    g.savefig('C://Users//jd-vz//Desktop//Code//src//explore//facet_pdf//' + f'{col}.pdf')
# %%
# * Transfers in out and balance against was home
import seaborn as sns
df = pd.read_csv(data_file)
for col in ['transfers_in', 'transfers_out', 'transfers_balance']:
    g = sns.FacetGrid(df, col="was_home", height=5, aspect=1)
    g.map(sns.histplot, col)

# %%
# * Create plot of value vs total points for all positions in
import seaborn as sns
df = pd.read_csv(data_file)
sns.jointplot(
    x='value',
    y='total_points',
    data=df,
    hue='position',
    kind='scatter',
    joint_kws={'alpha': 0.3})
# %%
# * Effect of fixture difficulty rating on premium_players
df = pd.read_csv(data_file)
team_stats = df[['team', 'strength_attack_home', 'strength_attack_away',
                    'strength_defence_home', 'strength_defence_away', 'season']].drop_duplicates()
team_stats.rename(columns={'team': 'opponent_team', 'strength_attack_home': 'opponent_strength_attack_home',
                            'strength_attack_away': 'opponent_strength_attack_away', 'strength_defence_home': 'opponent_strength_defence_home',
                            'strength_defence_away': 'opponent_strength_defence_away'}, inplace=True)
df = pd.merge(df, team_stats, on=['opponent_team', 'season'])
df['player_team_strength'] = np.where(df['was_home'], df['strength_attack_home'], df['strength_attack_away'])
df['player_team_defence'] = np.where(df['was_home'], df['strength_defence_home'], df['strength_defence_away'])
df['player_team_overall'] = df['player_team_strength'] / 2 + df['player_team_defence']/2
df['opponent_team_strength'] = np.where(df['was_home'], df['opponent_strength_attack_away'], df['opponent_strength_attack_home'])
df['opponent_team_defence'] = np.where(df['was_home'], df['opponent_strength_defence_away'], df['opponent_strength_defence_home'])
df['opponent_team_overall'] = df['opponent_team_strength'] / 2 + df['opponent_team_defence']/2

def quantile_function(df, quantile_point, col = 'value'):
    #Get the quantile value
    quantile_value = df.quantile(quantile_point)[col]
    #Select the data in the group that falls at or below the quantile value and return it
    return df[df[col] >= quantile_value]


df['FDR'] = np.where(df['opponent_team_overall'] > df['player_team_overall'] + 10, 'High',
                     np.where(df['opponent_team_overall'] < df['player_team_overall'] - 10, 'Low', 'Med')) 
df['premium_cutoff'] = df.groupby('position')['value'].transform('quantile', 0.75)
df['medium_cutoff'] = df.groupby('position')['value'].transform('quantile', 0.5)

df['premium_players'] = np.where(df['value'] >= df['premium_cutoff'], 'Premium', 
                                 np.where(df['value'] >= df['medium_cutoff'], 'Medium', 'Budget' ))
g = sns.lmplot(x="value", y="total_points", row_order=['Low', 'Med', 'High'], row="FDR", col="position", hue = 'premium_players', data=df)
df.groupby(['position', 'premium_players', 'was_home'])[['value','total_points']].corr().drop(['total_points'], axis = 1)
# Goalkeepers: Price is not a significant barrier to performance
# Midfielders: The strongest correlation, where you get what you pay for
# %%
# * Position, probability of winning, and value bins
df = pd.read_csv(data_file)
df['win'] = np.where(df['was_home'],
                     np.where(df['team_h_score'] > df['team_a_score'], 1, 0),
                     np.where(df['team_a_score'] > df['team_h_score'], 1, 0))


df['premium_cutoff'] = df.groupby('position')['value'].transform('quantile', 0.75)
df['medium_cutoff'] = df.groupby('position')['value'].transform('quantile', 0.5)
df['premium_players'] = np.where(df['value'] >= df['premium_cutoff'], 'Premium', 
                                 np.where(df['value'] >= df['medium_cutoff'], 'Medium', 'Budget' ))
g = sns.FacetGrid(df, col="position", hue = 'premium_players')
g.map(sns.kdeplot, 'win', cumulative=True, common_norm=False, common_grid=True)

# %%
df['was_home']
# %%
# * Mean cost vs value positions
df = pd.read_csv(data_file)
df['binned_cost'] = df.groupby('position').value.apply(pd.cut, bins=10)
df['mean_pts'] = df.groupby(['binned_cost', 'position'])['total_points'].transform('mean')
df['mean_value'] = df.groupby(['binned_cost', 'position'])['value'].transform('mean')
g = sns.lmplot(x="mean_value", y="mean_pts", col="position", col_order=['GK', 'DEF', 'MID', 'FWD'],  data=df)
# %%
# * Calculate correlation for all dataframes

# %%
import july
from july.utils import date_range
df = pd.read_csv(data_file, parse_dates=['kickoff_time'])
df.set_index('kickoff_time', inplace=True)
dates_1 = date_range("2019-01-01", "2019-12-31")
dates_2 = date_range("2020-01-01", "2020-12-31")
dates_3 = date_range("2021-01-01", "2021-12-31")
for date in [dates_1, dates_2, dates_3]:
    july.heatmap(
    date,
    df['total_points'], 
    cmap="github", 
    colorbar=True, 
    dpi = 500,
    title="Average points scored")

# %%
sns.set_style('darkgrid')
df = pd.read_csv(data_file)
abs(df.corrwith(df["total_points"])).sort_values(ascending=False).index[1:10]
# Ten highest correlated features
# Index(['bps', 'bonus', 'influence', 'goals_scored', 'npg', 'ict_index',
#        'clean_sheets', 'assists', 'xG'],
#       dtype='object')
import plotly.figure_factory as ff

figure = ff.create_scatterplotmatrix(
    df[['bps', 'bonus', 'influence', 'goals_scored', 'npg', 'ict_index', 'clean_sheets', 'assists', 'xG', 'position']],
    diag='histogram',
    index='position')
# %%
figure
# %%
df = pd.read_csv(data_file)

def corr(x, y, **kwargs):
    
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)
    
# Create a pair grid instance
grid = sns.PairGrid(data= df, vars = ['bps', 'bonus', 'influence', 'goals_scored', 'npg', 'ict_index', 'clean_sheets', 'assists', 'xG'], size = 5)

# Map the plots to the locations
grid = grid.map_upper(plt.scatter)
grid = grid.map_upper(corr)
grid = grid.map_lower(sns.kdeplot)
grid = grid.map_diag(plt.hist, bins = 30);
# %%
df.select_dtypes(include = 'number').sum()
# %%
import numpy as np
import pandas as pd
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_us.csv')
def outliers_removal(df, factor):
    for col in df.columns:
        Q1= df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + factor * IQR
        lower_limit = Q1 - factor * IQR
        if df[col].max() < upper_limit:
            print(f'{col} contains less than {factor} times the IQR of {Q3}')
   
outliers_removal(df.select_dtypes(include='number'), factor = 1)
# %%
for i in range(1,2):
    print(i)
# %%
import squarify

origin_counts = df.groupby('player_name').size().reset_index(name='counts')
sizes = origin_counts.counts.to_list()
color = plt.cm.Dark2(np.random.rand(len(sizes)))
label = list(zip(origin_counts.player_name, origin_counts.counts))
# treemap plot
plt.figure(figsize=(12,6))
squarify.plot(sizes=sizes, 
              color=color, 
              label=label, 
              pad=True)



# labels

plt.title('Treemap of Car Origins')
plt.axis('off')
# %%
