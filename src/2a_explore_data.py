# %%
# This script serves for the visual EDA
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib
sns.set(style="darkgrid")
sns.set(rc={"figure.figsize":(12, 6)}) 
plt.figure(figsize=(12, 6))
data_file =  'C://Users//jd-vz//Desktop//Code//data//collected_us.csv'
# %%
# * 1. The total team points scored in different seasons (Complete)
# * Note: Set season
sns.set(rc={"figure.figsize":(12, 6)}) 
df = pd.read_csv(data_file)
df = df[df['season'] == 2020] # Change season here
df['TeamTotalPointsCount'] = df.groupby(['team', 'season'])['total_points'].transform('sum')
df['TeamTotalCount'] = df.groupby(['team', 'season'])['total_points'].transform('sum')
df['TeamTotalCostCount'] = df.groupby(['team', 'season'])['value'].transform('sum')
df.sort_values('TeamTotalCount',ascending=False, inplace=True);
df = df[['team', 'TeamTotalPointsCount', 'TeamTotalCount','season', 'TeamTotalCostCount']].drop_duplicates()
ax = sns.barplot(x="team", y="TeamTotalPointsCount", data=df, ci=None, label = 'Total team points scored', palette =sns.color_palette('Blues_r', 40))
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
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//team_points_2020.pdf', bbox_inches='tight') 
# %%
# * 2. The total points scored by positions in different seasons (Complete)
# * Plus the effect of supporters on player performance
df = pd.read_csv(data_file)
ax_1 = sns.catplot(x="position", y="total_points",
                   hue="was_home", ci=90, col = 'season', col_order = [2019,2020], 
                   n_boot=2000, kind="point", data=df, palette = sns.color_palette(["#7e7e7e", "#69d"] ))
ax_1.set_xlabels('Position', fontsize=15) # not set_label
ax_1.set_ylabels('Average points scored', fontsize=15) # not set_label
ax_1._legend.texts[0].set_text("Away")
ax_1._legend.texts[1].set_text("Home")
ax_1._legend.set_title('')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//position_points_2019_2020.pdf', bbox_inches='tight') 
df['Supporters'] = np.where(df['kickoff_time'] < '2020-03-13', True, False)
ax_2 = sns.catplot(x="position", y="total_points", col="Supporters", col_order = [True, False],
                   hue="was_home", ci=90, n_boot=2000, kind="point", data=df,palette = sns.color_palette(["#7e7e7e", "#69d"] ))
ax_2.set_xlabels('Position', fontsize=15) # not set_label
ax_2.set_ylabels('Average points scored', fontsize=15) # not set_label
ax_2._legend.texts[0].set_text("Away")
ax_2._legend.texts[1].set_text("Home")
ax_2._legend.set_title('')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//Supporters.pdf', bbox_inches='tight') 
# df['Supporters'].value_counts()
# %%
# * 3. The total team and unique player counts scored by positions in different seasons (Complete)
sns.set(rc={"figure.figsize":(12, 6)}) 
df = pd.read_csv(data_file)
df['counter'] = 1
df['TeamTotalCount'] = df.groupby(['team'])['minutes'].transform('sum')
df.loc[df['TeamTotalCount'].idxmax()]
# %%
df.sort_values('TeamTotalCount',ascending=False, inplace=True);
ax = sns.barplot(x='team', y = 'TeamTotalCount',  data=df, label = 'Team', ci = None,  color = matplotlib.colors.to_hex(sns.color_palette('pastel')[0]))
plt.xticks(rotation=90)
plt.xlabel('Teams')
plt.ylabel('Minutes')
width_scale = 0.5
for bar in ax.containers[0]:
    bar.set_width(bar.get_width() * width_scale)
df['TeamPlayerCount'] = df.groupby(['team', 'player_name'])['counter'].transform('sum')
df['TeamTotalCount'] = df.groupby(['team'])['counter'].transform('sum')
df['player_count'] = df.groupby(['team'])['total_points'].transform('sum')
ax2 = ax.twinx()
ax_3 = sns.barplot(x='team', y = 'player_count', ci = None, label = 'Player', data=df,  
                   ax=ax2,  color = matplotlib.colors.to_hex(sns.color_palette('pastel')[1]))
for bar in ax2.containers[0]:
    x = bar.get_x()
    w = bar.get_width()
    bar.set_x(x + w * (1- width_scale))
    bar.set_width(w * width_scale)
plt.grid(False)
handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax_3.get_legend_handles_labels())]
# ax_3.legend(handles, labels, loc='upper right')
plt.ylabel('Points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//team_min_and_pt_counts.pdf', bbox_inches='tight') 
# %%
# * 4c) The number of times unique players played in gameweeks (Complete)
df = pd.read_csv(data_file)
df = df.groupby(['team', 'player_name', 'GW', 'season']).size().reset_index(name='counts') 
df['total_player_counts'] = df.groupby(['team', 'season', 'GW'])['counts'].transform('sum')
df.sort_values(['total_player_counts'], ascending=False, inplace=True)
ax = sns.catplot(x = 'team', y = 'total_player_counts',  kind = 'boxen', palette =sns.color_palette('Blues_r', 30),data = df, aspect = 3, height = 6)
plt.xlabel('Teams')
plt.xticks(rotation=90)
plt.ylabel('Non-unique players used in gameweeks')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//team_with_gameweek_counts.pdf', bbox_inches='tight') 
# ax._legend.set_title('Season')
# * Plus the reason why some are so high (Complete)
df = pd.read_csv(data_file)
df = df.groupby(['team', 'opponent_team', 'kickoff_time','player_name', 'GW', 'season', 'was_home', 'team_h_score', 'team_a_score']).size().reset_index(name='counts') 
df['total_player_counts'] = df.groupby(['team', 'season', 'GW'])['counts'].transform('sum')
df = df.loc[df.season == 2020,:]
df = df.loc[df.team == 'Man Utd',:]
df = df.loc[df.GW == 35,:]
print(df[['team', 'opponent_team','kickoff_time', 'was_home', 'team_h_score', 'team_a_score']].drop_duplicates())
print(len(df['player_name'].unique()), 'unique players for MUtd in GW 35')
# %%
# * Distribution of fixtures, and double and triple weeks for all teams (Complete)
df = pd.read_csv(data_file)
df = df[['team', 'GW', 'season','kickoff_time']].drop_duplicates()
df['counter'] = 1 # Num fixtures
df['num_matches'] = df.groupby(['team', 'GW', 'season'])['counter'].transform('sum') # Num matc hes played per gameweek by team
df['counter_matches'] = df.groupby(['team', 'num_matches'])['counter'].transform('sum') # Total matches played by team in both seasons
df = df[['team', 'num_matches', 'counter_matches']].drop_duplicates()
df = df.sort_values('counter_matches', ascending=False)
ax1 = sns.catplot(x = 'team', y = 'counter_matches', kind = 'bar', hue = 'num_matches', palette = 'pastel',
                  data = df,dodge = False,aspect = 2, height = 6)
plt.xlabel('Team')
plt.ylabel('Fixtures')
ax1._legend.set_title('Fixtures/Week')
plt.xticks(rotation=90)
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//num_double_weeks.pdf', bbox_inches='tight') 
# 19963 single weeks, 1106 double weeks, eighteen triple weeks

# %%
# * 4c) The number of times teams used positions in different matches (Complete)
# * Plus only one forward is required
df = pd.read_csv(data_file)
df = df.groupby(['position', 'team', 'kickoff_time', 'season']).size().reset_index(name='counts') 
df['total_position_counts'] = df.groupby(['position', 'team', 'kickoff_time'])['counts'].transform('sum')
df.sort_values(['total_position_counts'], ascending=False, inplace=True)
ax = sns.catplot(x = 'position', y = 'total_position_counts', 
                 hue = 'season', kind = 'boxen', data = df, aspect =2,
                 height = 6, palette = 'pastel')
plt.xlabel('Position')
plt.ylabel('Distribution of positions used in gameweeks')
ax._legend.set_title('Season')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//position_match_counts.pdf', bbox_inches='tight') 
# %%
# * 4e) The number of unique fixtures for all teams (Complete)
# seas = 2019
# df = df.loc[df['season'] == seas,:]
# print(df.loc[df.season == seas,['team', 'opponent_team','kickoff_time']].drop_duplicates().shape[0], 'fixtures') # 760 fixtures and each team plays 38 matches
# print(df.loc[df.season == seas,['team', 'opponent_team','kickoff_time']].drop_duplicates().shape[0]/len(df['team'].unique()), 'matches by each team') # 760 fixtures and each team plays 38 matches
# * Plus the number of unique fixtures each gameweek   (Complete)
sns.set(rc={"figure.figsize":(12, 6)}) 
df = pd.read_csv(data_file)
df = df[['player_name', 'kickoff_time', 'GW', 'season']].drop_duplicates()
df = df.groupby(['GW', 'season']).size().reset_index(name='Num_Matches_GW') 
# ax = sns.catplot(x="GW", y="Num_Matches_GW", hue='season', kind = 'bar', aspect = 2, height = 6, data=df, legend = True)
ax1 = sns.histplot(df, x='GW', hue='season', weights='Num_Matches_GW', multiple='stack',bins=38, palette='pastel', edgecolor = 'black')
plt.xlabel('Gameweek')
plt.ylabel('Players')
legend = ax1.get_legend()
handles = legend.legendHandles
legend.remove()
ax1.legend(handles, ['2019', '2020'], title='Season')
# plt.show()
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//player_counts.pdf', bbox_inches='tight') 
# %%
# * Plus a cumulative barplot showing the number of fixtures played each gameweek by teams  (Complete)
df = pd.read_csv(data_file)
df = df[['team', 'opponent_team','kickoff_time', 'GW', 'season']].drop_duplicates()
df = df.groupby(['GW', 'team', 'season']).size().reset_index(name='Num_Matches_GW') # No consideration of season with this way
plt.figure(figsize=(20, 8))
ax = sns.histplot(df, x='GW', hue='team',
                  weights='Num_Matches_GW', 
                  multiple='stack',bins=38, 
                  palette=sns.color_palette('tab20c', 23),legend=True,
                  cumulative = True, edgecolor='black') # tab20c
plt.xlabel('Gameweek')
plt.ylabel('Fixtures played')
legend = ax.get_legend()
handles = legend.legendHandles
legend.remove()
ax.legend(handles, df['team'].unique().ravel(), title='Team')
# plt.show()
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//fixtures_played.pdf', bbox_inches='tight') 
# %%
# *6) Histogram displaying distribution of kickoff times (Complete)
# * Plus suspended 13 March 2020 to 17 June
import matplotlib.dates as mdates
plt.figure(figsize=(15, 6))
df = pd.read_csv(data_file)
date_format = '%Y-%m-%d'
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], format=date_format)
# ax = sns.kdeplot(x = 'kickoff_time', data = df,hue='season',shade = True,cut = 0.8,clip_on=False,cumulative = False, fill=True, alpha=0.5,  palette ='pastel') 
ax = sns.histplot(x = 'kickoff_time', data = df,hue='season',  palette ='pastel', alpha = 1, bins = 40) 
fmt_half_year = mdates.WeekdayLocator(interval=2)
ax.xaxis.set_major_locator(fmt_half_year)
fmt_month = mdates.WeekdayLocator(interval=2)
ax.xaxis.set_minor_locator(fmt_month)
ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
ax.format_xdata = mdates.DateFormatter(date_format)
ax.grid(False)
plt.xticks(rotation=45)
plt.xlabel('Kickoff Time')
plt.ylabel('Density')
legend = ax.get_legend()
handles = legend.legendHandles
legend.remove()
ax.legend(handles, ['2019', '2020'], loc = 'upper left',title='Season')
# plt.show()
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//distirb_kickoff_times.pdf', bbox_inches='tight') 
print(min(df.loc[df['season'] == 2020, 'kickoff_time']))
print(max(df.loc[df['season'] == 2019, 'kickoff_time']))
# %%
# *6a) Plot ploints scored in years(Complete)
df = pd.read_csv(data_file)
plt.figure(figsize=(12, 6))
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
df = df.set_index('kickoff_time')
df['Year'] = df.index.year
sns.boxplot(data=df, x = 'Year', y='total_points', palette ='pastel') 
plt.xlabel('Year')
plt.ylabel('Points scored')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//points_years.pdf', bbox_inches='tight') 
# *6b) Plot ploints scored in months (Complete)
plt.figure(figsize=(15, 6))
df['Month'] = df.index.month
sns.boxplot(data=df, x = 'Month', y='total_points', palette ='pastel') 
plt.xlabel('Month')
plt.ylabel('Points scored')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//points_monts.pdf', bbox_inches='tight') 
# *6c) Plot ploints scored in days (Complete)
plt.figure(figsize=(12, 6))
df['Weekday Name'] = df.index.day_name()
sns.boxplot(data=df, x = 'Weekday Name', y='total_points', order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], palette ='pastel') 
plt.xlabel('Weekday')
plt.ylabel('Points scored')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//points_days.pdf', bbox_inches='tight') 
# * Mean plots in gameweeks series plot of points scored (Complete)
ax1 = sns.catplot(x = 'GW', y = 'total_points', kind = 'bar', ci = None, hue = 'season', palette = 'pastel',data = df,dodge = True, aspect = 2, height = 6)
plt.xlabel('Gameweek')
plt.ylabel('Points scored')
ax1._legend.set_title('Season')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//points_weeks.pdf', bbox_inches='tight') 
# %%
# * Plots player cost vs number of times selected (Complete)
import matplotlib
plt.figure(figsize=(12, 6))
df = pd.read_csv(data_file)
df['player_season_selected'] = df.groupby(['player_name', 'season'])['selected'].transform('sum') # All player points in a season
df['player_season_value'] = df.groupby(['player_name', 'season'])['value'].transform('sum') # All player points in a season
df = df[['player_name','season', 'player_season_selected', 'player_season_value', 'position']].drop_duplicates()
df_temp = df.sort_values(['player_season_selected'],ascending=False).head(5)
df = pd.read_csv(data_file)
df = df.loc[df['player_name'].isin(df_temp['player_name'].unique())]
ax = sns.barplot(x="GW", y="selected", data=df, ci=None, label = 'Selected', color = 'gray')
# ax = sns.pointplot(x="GW", y="selected", data=df, ci=None, label = 'Selected', color =  matplotlib.colors.to_hex(sns.color_palette('pastel')[0]))
plt.xlabel('Gameweek')
plt.ylabel('Selected')
plt.grid(False)
# width_scale = 0.5
# for bar in ax.containers[0]:
#     bar.set_width(bar.get_width() * width_scale)
ax2 = ax.twinx()
ax_3 = sns.barplot(x='GW', y='value', label = 'Value', data=df, ax = ax2,ci=None, color = matplotlib.colors.to_hex(sns.color_palette('pastel')[1]))
# ax_3 = sns.pointplot(x='GW', y='value', label = 'Value', data=df, ax = ax2,ci=None, color =  matplotlib.colors.to_hex(sns.color_palette('pastel')[0]))
ax_3.set(ylim=(80, 110))
plt.ylabel('Value')
# for bar in ax2.containers[0]:
#     x = bar.get_x()
#     w = bar.get_width()
#     bar.set_x(x + w * (1- width_scale))
#     bar.set_width(w * width_scale)
plt.grid(False)
handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax_3.get_legend_handles_labels())]
ax_3.legend(handles, labels, loc='upper left')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//selected_vs_value.pdf', bbox_inches='tight')
# %%
# * Mean points achieved by positions for all common choices with times  (Complete)
df = pd.read_csv(data_file)
df['selected_qnt'] = df.groupby(['GW','position'])['selected'].transform('quantile', 0.8)
df['Common transfer'] = np.where(df['selected'] > df['selected_qnt'], True, False)
df['Time played'] = np.where(df['minutes'] > 45, '(45, 90]',
                             np.where(df['minutes'] > 10, '(10, 45]', '[0, 10]'))
df['Common transfer'] = np.where(df['selected'] > df['selected_qnt'], True, False)
ax_2 = sns.catplot(x="position", y="total_points", col = 'Time played', 
                   col_order=['[0, 10]', '(10, 45]', '(45, 90]'],
                   hue="Common transfer", ci=None, kind="bar", data=df,
                   palette = 'pastel')
ax_2.set_xlabels('Position', fontsize=15) 
ax_2.set_ylabels('Average points', fontsize=15) 
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//commonly_selected_points.pdf', bbox_inches='tight') 
# %%
# * Binned minutes vs distribution cumulative density (Complete)
df = pd.read_csv(data_file)
plt.figure(figsize=(12, 6))
BIN_VALUE = 4
df['minutes_bins'] = pd.cut(df['minutes'], bins=BIN_VALUE,  labels = ['[0, 22.5]', '(22.5, 45]', '(45, 67.5]', '(67.5,90]'])
ax = sns.kdeplot(data=df, x="total_points", hue="minutes_bins", palette = 'inferno',
            fill = True, alpha = 0.09, cumulative = False,legend=True,
            common_norm=True, common_grid=False)
legend = ax.get_legend()
handles = legend.legendHandles
legend.remove()
ax.legend(handles,['[0, 22.5]', '(22.5, 45]', '(45, 67.5]', '(67.5,90]'], loc = 'upper right',title='Minutes played')
plt.xlabel('Total points scored')
plt.ylabel('Cumulative density')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//minutes_points.pdf', bbox_inches='tight') 
# * For > 1 points (partticipation)
df = pd.read_csv(data_file)
plt.figure(figsize=(12, 6))
BIN_VALUE = 4
df['minutes_bins'] = pd.cut(df['minutes'], bins=BIN_VALUE,  labels = ['[0, 22.5]', '(22.5, 45]', '(45, 67.5]', '(67.5,90]'])
ax = sns.kdeplot(data=df[df['total_points'] > 1], x="total_points", hue="minutes_bins", palette = 'inferno',
            fill = True, alpha = 0.09, cumulative = False,legend=True,
            common_norm=True, common_grid=False)
legend = ax.get_legend()
handles = legend.legendHandles
legend.remove()
ax.legend(handles,['[0, 22.5]', '(22.5, 45]', '(45, 67.5]', '(67.5,90]'], loc = 'upper left',title='Minutes played')
plt.xlabel('Total points scored')
plt.ylabel('Cumulative density')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//minutes_points_above_1_pt.pdf', bbox_inches='tight') 
# %%
# * Transfer anomalies (Complete)
import matplotlib
plt.figure(figsize=(12, 6))
df = pd.read_csv(data_file)
df = df[df['season'] == 2020]
df = df[df['player_name'] == df['player_name'].unique()[2]] # df = pd.read_csv(data_file)
ax =sns.lineplot(x = df['GW'], y = df['transfers_in'],
              label='Transfers in', color = matplotlib.colors.to_hex(sns.color_palette('pastel')[2]))
sns.lineplot(x = df['GW'],
              y = df['transfers_out'],
              label='Transfers out',  color = matplotlib.colors.to_hex(sns.color_palette('pastel')[3]))
sns.lineplot(x = df['GW'], y = df['transfers_balance'], 
              label='Transfers balance',  color = matplotlib.colors.to_hex(sns.color_palette('pastel')[1]))
plt.fill_between(df['GW'].values, df['transfers_in'].values,  alpha = 0.4,   color = matplotlib.colors.to_hex(sns.color_palette('pastel')[2]))
plt.fill_between(df['GW'].values, df['transfers_in'].values, df['transfers_out'].values, alpha = 0.4,  color = matplotlib.colors.to_hex(sns.color_palette('pastel')[3]))
plt.fill_between(df['GW'].values, df['GW'].values, df['transfers_balance'].values, alpha = 0.4,  color = matplotlib.colors.to_hex(sns.color_palette('pastel')[1]))
legend = ax.legend()
plt.xlabel('Gameweek')
plt.ylabel('Transfers')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//transfers_gameweek.pdf', bbox_inches='tight') 
# %%
# *7 Plot the top fifty scoring players with their associated cost (Complete)
# * Plus we aim to maximize ROI
from matplotlib.lines import Line2D
plt.figure(figsize=(15, 8))
sns.set_style('darkgrid')
df = pd.read_csv(data_file)
df = df[df['season'] == 2019]
df['player_season_points'] = df.groupby(['player_name', 'season'])['total_points'].transform('sum') # All player points in a season
df['player_season_cost'] = df.groupby(['player_name', 'season'])['value'].transform('mean') # Mean cost throughout the season
df = df[['player_name', 'player_season_points', 'player_season_cost',  'position']].drop_duplicates()

NUM = 15
df_1 = df[df['position'] == 'FWD'].sort_values('player_season_points', ascending = False).head(NUM)
df_2 = df[df['position'] == 'MID'].sort_values('player_season_points', ascending = False).head(NUM)
df_3 = df[df['position'] == 'GK'].sort_values('player_season_points', ascending = False).head(NUM)
df_4 = df[df['position'] == 'DEF'].sort_values('player_season_points', ascending = False).head(NUM)
df_temp = pd.concat([df_2,  df_1, df_4, df_3])

palette = [color for color in [[matplotlib.colors.to_hex(sns.color_palette('YlOrBr_r', 20)[i]) for i in range(NUM)],
                               [matplotlib.colors.to_hex(sns.color_palette('YlGn_r', 20)[i]) for i in range(NUM)],
                               [matplotlib.colors.to_hex(sns.color_palette('PuBu_r', 20)[i]) for i in range(NUM)],
                               [matplotlib.colors.to_hex(sns.color_palette('PiYG', 40)[i]) for i in range(NUM)]]] 
flt = list(np.array(palette).ravel())
    
ax = sns.barplot(x='player_name', y='player_season_points', label = 'Return on investment for season', data = df_temp, ci=None, palette=flt, edgecolor = 'black')
sns.barplot(x='player_name', y='player_season_cost', data = df_temp, ci=None, alpha = 0.3, color = 'white', edgecolor ='black')
plt.xticks(rotation=90)
plt.xlabel('Player name')
plt.ylabel('Player season points') 
plt.grid(False)

colors = [sns.color_palette('YlOrBr_r', 20)[0],
          sns.color_palette('YlGn_r', 20)[0],
          sns.color_palette('PuBu_r', 20)[0],
          sns.color_palette('PiYG', 40)[0]]

legend_elements = [Line2D([0], [0], color=colors[0], lw=2, label='MID'), # MID, FWD, DEF, GK, 2,1,4,3
                   Line2D([0], [0], color=colors[1], lw=2, label='FWD'),
                   Line2D([0], [0], color=colors[2], lw=2, label='DEF'), 
                   Line2D([0], [0], color=colors[3], lw=2, label='GK')]

ax.legend(handles=legend_elements, loc='upper right',  title = 'Position')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//top_15_positions_pts_2019.pdf', bbox_inches='tight') 

# %%
# * 8 Top fifty players sorted by their return on investment (Complete)
import matplotlib
from matplotlib.lines import Line2D
df = pd.read_csv(data_file)
df = df[df['season'] == 2019]
plt.figure(figsize=(15, 8))
df['player_season_points'] = df.groupby(['player_name'])['total_points'].transform('sum') # All player points in a season
df['player_season_cost'] = df.groupby(['player_name'])['value'].transform('mean') # Mean cost throughout the season
df['player_season_roi'] = df['player_season_points'] / df['player_season_cost']
df = df[['player_name', 'player_season_roi', 'position']].drop_duplicates()
df = df.sort_values(['player_season_roi'],ascending=False)
NUM = 15
df_1 = df[df['position'] == 'FWD'].head(NUM)
df_2 = df[df['position'] == 'MID'].head(NUM)
df_3 = df[df['position'] == 'GK'].head(NUM)
df_4 = df[df['position'] == 'DEF'].head(NUM)
df_temp = pd.concat([df_3, df_4,df_1, df_2])
palette = [color for color in [[matplotlib.colors.to_hex(sns.color_palette('PiYG', 40)[i]) for i in range(NUM)], 
                               [matplotlib.colors.to_hex(sns.color_palette('PuBu_r', 20)[i]) for i in range(NUM)], 
                               [matplotlib.colors.to_hex(sns.color_palette('YlOrBr_r', 20)[i]) for i in range(NUM)],
                               [matplotlib.colors.to_hex(sns.color_palette('YlGn_r', 20)[i]) for i in range(NUM)]]]
flt = list(np.array(palette).ravel())
ax = sns.barplot(x='player_name', y='player_season_roi', label = 'Return on investment for season', data = df_temp, ci=None, palette=flt,  edgecolor = 'black')
plt.xticks(rotation=90)
plt.xlabel('Player name')
plt.ylabel('Return on investment') 
plt.grid(False)

colors = [sns.color_palette('YlOrBr_r', 20)[0],
          sns.color_palette('YlGn_r', 20)[0],
          sns.color_palette('PuBu_r', 20)[0],
          sns.color_palette('PiYG', 40)[0]]

legend_elements = [Line2D([0], [0], color=colors[0], lw=2, label='FWD'), # MID, FWD, DEF, GK, 2,1,4,3 Old order
                   Line2D([0], [0], color=colors[1], lw=2, label='MID'), # GK, DEF, MID, FWD New order 3,4,2,1
                   Line2D([0], [0], color=colors[2], lw=2, label='DEF'), 
                   Line2D([0], [0], color=colors[3], lw=2, label='GK')]

ax.legend(handles=legend_elements, loc='upper right', title = 'Position')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//player_season_roi.pdf', bbox_inches='tight') 
# %%
# * 8 Value versus total points (Complete)
import seaborn as sns
sns.set_style('darkgrid')
plt.figure(figsize=(15, 8))
pal = sns.dark_palette("#69d", reverse=True, as_cmap=False,n_colors=150)
df = pd.read_csv(data_file)
BIN_VALUE = 4
df['Position'] = df['position']
ax = sns.jointplot(x = 'value', y = 'total_points', hue = 'Position', palette = sns.color_palette('deep', 4),
                data = df,  edgecolor="gray", alpha = 0.8, height = 10)
ax.set_axis_labels('Value', 'Points', fontsize=16)


plt.savefig('C://Users//jd-vz//Desktop//Code//fig//positions_value_vs_total_points.png', bbox_inches='tight') 
# %%
# %%
# * Plots ppm to value(Complete)
df = pd.read_csv(data_file)
plt.figure(figsize=(12, 6))
df['player_points'] = df.groupby(['player_name','season'])['total_points'].transform('sum') # Cut of point for bottom 90 and top 10 of each gameweek
df['player_value'] = df.groupby(['player_name','season'])['value'].transform('mean') # Cut of point for bottom 90 and top 10 of each gameweek
df['player_points_per_million_cost'] = df['player_points'] / df['player_value'] 
ax1 = sns.scatterplot(x = 'player_points', y = 'player_points_per_million_cost', 
                      data = df, hue = 'position')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc='upper left').set_title('Position')
plt.xlabel('Player season points')
plt.ylabel('Player points to value')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//ppm_value.pdf', bbox_inches='tight') 
# %%
# * 10 Plot some position statistics (Complete)
df = pd.read_csv(data_file)
fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows =1, sharey=False, sharex = False)
sns.set()
df = df[['position', 'influence',  'bps', 'threat', 'creativity', 'ict_index', 'total_points']]
df.groupby('position').sum().T.plot(kind='bar', stacked=True,ax = ax1)
ax1.get_legend().remove()
df = pd.read_csv(data_file)
df['Position'] = df['position']
df = df[['Position', 'xGChain', 'xGBuildup', 'xG', 'npxG', 'npg',   'xA']]
df.groupby('Position').sum().T.plot(kind='bar', stacked=True, ax = ax2)
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//position_stats_bar.pdf', bbox_inches='tight') 
# %%
# * 10 Plot some position understat statistics (Complete)
sns.set(style='darkgrid')
plt.figure(figsize=(12,6))
df = pd.read_csv(data_file)
df = df[['position', 'xGChain', 'xGBuildup', 'xG', 'npxG', 'npg',   'xA']]
df.groupby('position').sum().T.plot(kind='bar', stacked=True)
plt.legend(title = 'Position')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//position_understat_bar.pdf', bbox_inches='tight') 
# %%
# * 10 Plot goals vs xG measures (Complete)
df = pd.read_csv(data_file)
sns.pointplot(x = 'goals_scored', y = 'xG',
                  data = df,  ci = None,  hue = 'position',
                 legend = True)
plt.legend(title = 'Position')
plt.xlabel('Goals scored')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//xg_goals.pdf', bbox_inches='tight') 
# %%
# * 10 Plot npg vs npxG measures (Complete)
sns.pointplot(x = 'npg', y = 'npxG',
                  data = df,  ci = None,  hue = 'position',
                 legend = True)
plt.legend(title = 'Position')
plt.xlabel('Non-penalty goals')
plt.legend(title = 'Position')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//npg_npxG.pdf', bbox_inches='tight') 
# %%
# * 10 Plot assists vs xA measures (Complete)
sns.pointplot(x = 'assists', y = 'xA',
                  data = df,  ci = None,  hue = 'position',
                 legend = True)
plt.xlabel('Assists')
plt.legend(title = 'Position')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//assists_xA.pdf', bbox_inches='tight') 
# %%
# * 10 Plot assists vs key passes measures (Complete)
sns.pointplot(x = 'assists', y = 'key_passes',
                  data = df,  ci = None,  hue = 'position',
                 legend = True)
plt.xlabel('Assists')
plt.ylabel('Key passes')
plt.legend(title = 'Position')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//assists_key_passes.pdf', bbox_inches='tight') 
# %%
# * 10 Plot goals_scored vs shots taken (Complete)
sns.pointplot(x = 'goals_scored', y = 'shots',
                  data = df,  ci = None,  hue = 'position',
                 legend = True)
plt.xlabel('Goals scored')
plt.ylabel('Shots taken')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//goals_to_shots.pdf', bbox_inches='tight') 
# %%
# * Plot distribution of positions
df = pd.read_csv(data_file)
df['Position'] = df['position']
ax = sns.kdeplot(data=df[(df['total_points'] > 1) & (df['total_points'] < 5)], x="total_points", hue="position", palette = 'pastel',
            fill = True, alpha = 0.09, cumulative = False ,legend=True,
            common_norm=True, common_grid=False)
# %%
# * Understat vs fantasy positions
# Create attacking and defending midfielders
def position_stats(df):
    df.loc[df['position'] == 'GK','position_stat'].value_counts() # Marginal 
    df.loc[df['position'] == 'GK','position_role'] = 'GK'
    df.loc[df['position'] == 'FWD','position_stat'].value_counts() # Marginal
    df.loc[df['position'] == 'FWD','position_role'] = 'FWD'
    df.loc[df['position'] == 'DEF','position_stat'].value_counts() # Marginal
    df.loc[df['position'] == 'DEF','position_role'] = 'DEF'
    df.loc[df['position'] == 'MID','position_stat'].value_counts() # Potential for dividing here
    df.loc[df['position'] == 'MID','position_role'] = 'DMID' 
    df['str_cut'] = df['position_stat'].astype(str).str[0]
    df_temp  = df.loc[df['position'] == 'MID']
    def_names = df_temp.loc[df_temp['str_cut'] == 'D','player_name'].unique()
    attack_names = df_temp.loc[df_temp['str_cut'] == 'A','player_name'].unique()
    df.loc[df['player_name'].isin(def_names) & (df['position'] == 'MID'), 'position_role'] = 'DMID'
    df.loc[df['player_name'].isin(attack_names) & (df['position'] == 'MID'), 'position_role'] = 'AMID'
    # Create substitution boolean
    df['Sub'] = np.where(df['position_stat'] == 'Sub', 1,0)
    df['position_stat'].value_counts()
    # Create position location
    df['str_cut_1'] = df['position_stat'].astype(str).str[1]
    df['str_cut_2'] = df['position_stat'].astype(str).str[2]
    df['position_location'] = 'General'
    for feat in ['str_cut_1', 'str_cut_2']:
        df.loc[df[feat] == 'C', 'position_location'] = 'Centre'
        df.loc[df[feat] == 'R', 'position_location'] = 'Right'
        df.loc[df[feat] == 'L', 'position_location'] = 'Left'
    df = df.drop(['str_cut_1', 'str_cut_2'], axis = 1)
    return df

# *- Plot positions vs understat positions
df = pd.read_csv(data_file) 
df = position_stats(df)
df['Detailed Positions'] = df['position_stat']
sns.displot(df, x='position', hue='Detailed Positions', multiple='stack',height = 6,aspect = 1)
plt.xlabel('Position')
plt.ylabel('Count') 
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//position_to_us_position.pdf', bbox_inches='tight') 
# %%
# * Plot positions vs total points
df = pd.read_csv(data_file) 
df = position_stats(df)
df['Position'] = df['position']
sns.histplot(x = 'total_points', hue = 'Position', data = df,
            multiple="stack", palette='pastel', edgecolor=".3", bins=32, alpha = 0.7)
plt.xlabel('Total points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//position_vs_points_hist.pdf', bbox_inches='tight') 
plt.show()
sns.histplot(x = 'total_points', hue = 'Position', data = df[df['total_points'] < 0],
            multiple="stack", palette='pastel', edgecolor=".3", bins=4, alpha = 0.7)
plt.xlabel('Total points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//position_vs_points_hist_lt0.pdf', bbox_inches='tight') 
plt.show()
sns.histplot(x = 'total_points', hue = 'Position', data = df[(df['total_points'] > 1) & (df['total_points'] < 10)],
            multiple="stack", palette='pastel', edgecolor=".3", bins=8, alpha = 0.7)
plt.xlabel('Total points > 3')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//position_vs_points_hist_gt1.pdf', bbox_inches='tight') 
plt.show()
sns.histplot(x = 'total_points', hue = 'Position', data = df[df['total_points'] > 10],
            multiple="stack", palette='pastel', edgecolor=".3", bins=14, alpha = 0.7)
plt.xlabel('Total points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//position_vs_points_hist_gt10.pdf', bbox_inches='tight') 
plt.show()
df = df[df['position_role'].isin(['AMID', 'DMID'])]
df['Position'] = df['position_role']
sns.histplot(x = 'total_points', hue = 'Position', data = df,
            multiple="stack", palette='pastel', edgecolor=".3", bins=30, alpha = 0.7)
plt.xlabel('Total points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//position_vs_amiddmid.pdf', bbox_inches='tight') 
plt.show()
sns.histplot(x = 'total_points', hue = 'Position', data = df[df['total_points'] > 0],
            multiple="stack", palette='pastel', edgecolor=".3", bins=9, alpha = 0.7)
plt.xlabel('Total points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//position_vs_amiddmid_gt_0.pdf', bbox_inches='tight') 
# %%
# * Distribution of detailed positions to points
df = pd.read_csv(data_file) 
df = position_stats(df)
df['Detailed positions'] = df['position_stat']
plt.figure(figsize=(12, 6))
sns.kdeplot(x = 'total_points', hue = 'Detailed positions', data = df,
            multiple="stack", palette='pastel', edgecolor=".3",  alpha = 0.7)
plt.xlabel('Total points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//understat_positions_points.pdf', bbox_inches='tight') 
# %%
# * Distribution of fantasy positions to points
df = pd.read_csv(data_file) 
df['Positions'] = df['position']
plt.figure(figsize=(12, 6))
sns.kdeplot(x = 'total_points', hue = 'Positions', data = df,
            multiple="stack", palette='pastel', edgecolor=".3",  alpha = 0.7)
plt.xlabel('Total points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//fantasy_positions_points.pdf', bbox_inches='tight') 
# %%
# * Distribution of attacking and defending midfielders
df = pd.read_csv(data_file) 
df = position_stats(df)
df = df[df['position_role'].isin(['AMID', 'DMID'])]
df['Midfielders'] = df['position_role']
plt.figure(figsize=(12, 6))
sns.kdeplot(x = 'total_points', hue = 'Midfielders', data = df[df['total_points'] > 5],
            multiple="stack", palette='pastel', edgecolor=".3",  alpha = 0.7)
plt.xlabel('Total points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//points_to_mids.pdf', bbox_inches='tight') 
# %% 
# * Distribution of position locations
df = pd.read_csv(data_file) 
df = position_stats(df)
df['Location'] = df['position_location']
plt.figure(figsize=(12, 6))
sns.kdeplot(x = 'total_points', hue = 'Location', data =df[df['total_points'] > 5],
            multiple="stack", palette='pastel', edgecolor=".3",  alpha = 0.7)
plt.xlabel('Total points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//points_to_location.pdf', bbox_inches='tight') 
# %%
# * Plot attacking vs defneding midfielders
df = pd.read_csv(data_file) 
df = position_stats(df)
df = df[df['position_role'].isin(['AMID', 'DMID'])]
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
df['Midfielder'] = df['position_role']
df['Total points'] = df['total_points']
df['FDR'] = np.where(df['opponent_team_overall'] > df['player_team_overall'] + 10, 'High',
                     np.where(df['opponent_team_overall'] < df['player_team_overall'] - 10, 'Low', 'Med')) 
df['premium_cutoff'] = df.groupby('Midfielder')['value'].transform('quantile', 0.75)
df['medium_cutoff'] = df.groupby('Midfielder')['value'].transform('quantile', 0.5)
df['premium_players'] = np.where(df['value'] >= df['premium_cutoff'], 'Premium', 
                                 np.where(df['value'] >= df['medium_cutoff'], 'Medium', 'Budget' ))
df['Cost bracket'] = np.where(df['value'] >= df['premium_cutoff'], 'Premium', 
                                 np.where(df['value'] >= df['medium_cutoff'], 'Medium', 'Budget' ))
g = sns.FacetGrid(df, col="Midfielder", row = 'FDR',
                  row_order = ['Low', 'Med', 'High'], 
                  col_order = ['DMID', 'AMID'])
g.map(sns.pointplot, "Cost bracket", "Total points",'season', ci=None, order = ['Budget', 'Medium', 'Premium'],
      legend_out=True, hue_order = [2019,2020],palette = 'pastel')
g.add_legend(title = 'Season')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//att_def_mid.pdf', bbox_inches='tight') 
# %%
# * Plot centre, right, and general
# * Plot centre, right, and general
df['Location'] = df['position_location']
g = sns.FacetGrid(df,  col = 'FDR', col_order = ['Low', 'Med', 'High'])
g.map(sns.pointplot, "Location", "Total points", 'season', ci=None, 
      order = ['General', 'Centre', 'Left', 'Right'],
      legend_out=True, hue_order = [2019,2020],palette = 'magma')
g.add_legend(title = 'Season')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//player_field_positions.pdf', bbox_inches='tight')
# %%
# * Plot subsitution vs total points
df['Substitution'] = df['Sub']
df.loc[df['Substitution'] == 1, 'Substitution'] = True
df.loc[df['Substitution'] == 0, 'Substitution'] = False
sns.kdeplot('total_points', hue = 'Substitution', data = df, multiple="stack", palette='pastel', alpha = 0.7, edgecolor = 'gray')
plt.xlabel('Total points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//subs_to_pts.pdf', bbox_inches='tight') 
# %%
# * Plot subsitution vs total points
df = pd.read_csv(data_file) 
df = position_stats(df)
df = df[df['total_points'] > 5]
df['Location'] = df['position_location']
sns.kdeplot('total_points', hue = 'Location', data = df, multiple="layer", fill = False, palette='inferno', alpha = 0.5, cumulative = False)
# sns.histplot(x = 'total_points', hue = 'Location', data = df, multiple="stack", hue_order= ['Left', 'Right', 'Centre', 'General'], 
#              palette='pastel', edgecolor=".3", bins=14, alpha = 0.7)
plt.xlabel('Total points')

#  %%
# * Effect of fixture difficulty rating on premium_players (Complete)
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
g = sns.FacetGrid(df, col="position", row = 'FDR',
                  row_order = ['Low', 'Med', 'High'], 
                  col_order = ['GK', 'DEF', 'MID', 'FWD'])
g.map(sns.pointplot, "premium_players", "total_points",'season', ci=None, order = ['Budget', 'Medium', 'Premium'],
      legend_out=True, hue_order = [2019,2020],palette = 'magma')
g.add_legend(title = 'Season')


# Iterate thorugh each axis
i = 0
for ax in g.axes.flat:
    i = i + 1
    if i in [1,5,9]:
        ax.set_ylabel('Total points')
    ax.set_xlabel('')
    if ax.texts:
        # This contains the right ylabel text
        txt = ax.texts[0]
        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                txt.get_text().split('=')[1],
                transform=ax.transAxes,
                va='center')
        # Remove the original text
        ax.texts[0].remove()
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//premium_points.pdf', bbox_inches='tight') 
df.groupby(['position', 'premium_players', 'was_home'])[['value','total_points']].corr().drop(['total_points'], axis = 1)
# Goalkeepers: Price is not a significant barrier to performance
# Midfielders: The strongest correlation, where you get what you pay for

# %%
# * Position, probability of winning, and value bins (Complete)
df = pd.read_csv(data_file)
df['win'] = np.where(df['was_home'],
                     np.where(df['team_h_score'] > df['team_a_score'], 'Win', 'Lose'),
                     np.where(df['team_a_score'] > df['team_h_score'], 'Win', 'Lose'))
df['premium_cutoff'] = df.groupby('position')['value'].transform('quantile', 0.75)
df['medium_cutoff'] = df.groupby('position')['value'].transform('quantile', 0.5)
df['premium_players'] = np.where(df['value'] >= df['premium_cutoff'], 'Premium', 
                                 np.where(df['value'] >= df['medium_cutoff'], 'Medium', 'Budget' ))
df['Season'] = df['season']
df['Position'] = df['position']
ax_2 = sns.catplot(x = 'win', y = 'total_points', data = df, 
                  hue = 'premium_players', col = 'Position',
                  kind = 'bar', hue_order=['Budget', 'Medium', 'Premium'],
                  col_order = ['GK', 'DEF', 'MID', 'FWD'], ci = None, palette = 'pastel')
ax_2.set_ylabels('Average points scored')
ax_2.set_xlabels('')
ax_2._legend.set_title('Cost category')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//premium_wins.pdf', bbox_inches='tight') 
# %%
# *4f) Clustered heatmap of features (Complete)
# Multicollinearity: Spreading the same information across several predictors
df = pd.read_csv(data_file)
sns.set_theme()
print(df.corrwith(df["total_points"]).abs().sort_values(ascending = False).head(10)) # 0.95 with goals scored
sns.clustermap(df.select_dtypes(include ='number').corr(), figsize=(20,20))
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//correlations.pdf', bbox_inches='tight') 
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
print(f'Threshold {threshold}: columns that are collinear with other columns:', cols_to_remove)
#? 1. The team strength statistics are highly correlated --> Create FDR and remove
#? 2. Non penalty goals and goals scored are highly correlated --> Create penalty goal scored boolean
#? 3. Transfers balance can be dropped.
# %%
# * Collinear features (non-respones) (Complete)
df = pd.read_csv(data_file)
from feature_selector import *
fs = FeatureSelector(data = df.drop('total_points', axis=1), labels = df['total_points'])
fs.identify_collinear(correlation_threshold = 0.80, one_hot = False)
fs.plot_collinear()
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//collinear.pdf', bbox_inches='tight') 
# dataframe of collinear features
fs.record_collinear.head()
fs.ops['collinear']
df = pd.read_csv(data_file)
print(df.corrwith(df["total_points"]).abs().sort_values(ascending = False).head(10)) # 0.95 with goals scored
# %%
# * Brief correlation analysis
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
scale = StandardScaler()
df = pd.read_csv(data_file)
df = pd.DataFrame(scale.fit_transform(df.select_dtypes(include = 'number')),
                  columns=df.select_dtypes(include = 'number').columns, index=df.index)

def calculate_vif(data):
    vif_df = pd.DataFrame(columns = ['Var', 'Vif'])
    x_var_names = data.columns
    for i in range(0, x_var_names.shape[0]):
        y = data[x_var_names[i]]
        x = data[x_var_names.drop([x_var_names[i]])]
        r_squared = sm.OLS(y,x).fit().rsquared
        vif = round(1/(1-r_squared),2)
        vif_df.loc[i] = [x_var_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis = 0, ascending=False, inplace=False)

def get_VIF(dataFrame , target):
    X = add_constant(dataFrame.loc[:, dataFrame.columns != target])
    seriesObject = pd.Series([variance_inflation_factor(X.values,i) for i in range(X.shape[1])] , index=X.columns,)
    return seriesObject.sort_values(ascending = False)

print(get_VIF(df,'total_points'))
print(calculate_vif(df.select_dtypes(include = 'number').drop('total_points', axis = 1)))
print(df.head(5))
print(df.corrwith(df["key_passes"]).abs().sort_values(ascending = False).head(5)) # 0.96 correlation with creativity
print(df.corrwith(df["npg"]).abs().sort_values(ascending = False).head(5)) # 0.95 with goals scored
print(df.corrwith(df["npxG"]).abs().sort_values(ascending = False).head(5)) # 0.94 correlated with xG

# %%
# * Groupwise highest correlation with total points
df = pd.read_csv(data_file)
for col in ['season', 'clean_sheets', 'own_goals', 'penalties_missed', 'red_cards', 'yellow_cards']:
    df[col] = df[col].astype('category')
print(df[df['position'] == 'GK'].corrwith(df["total_points"]).abs().sort_values(ascending = False).head(5)) # BPS, Bonus, Clean sheets, influence
print(df[df['position'] == 'FWD'].corrwith(df["total_points"]).abs().sort_values(ascending = False).head(5)) # Clean sheets
print(df[df['position'] == 'DEF'].corrwith(df["total_points"]).abs().sort_values(ascending = False).head(5)) # Bonus
print(df[df['position'] == 'MID'].corrwith(df["total_points"]).abs().sort_values(ascending = False).head(5)) 
# %%
# * Plot of top scoring playter in different positions (Complete)
import plotly.express as px
import numpy as np
df = pd.read_csv(data_file)
plt.figure(figsize=(15, 6))
df['player_season_points'] = df.groupby(['player_name', 'season'])['total_points'].transform('sum') # All player points in a season
df['player_season_cost'] = df.groupby(['player_name', 'season'])['value'].transform('mean') # Mean cost throughout the season
df['player_season_minutes'] = df.groupby(['player_name', 'season'])['minutes'].transform('mean') # All minutes played throughout the season
df['player_season_bps'] = df.groupby(['player_name', 'season'])['bps'].transform('mean') # Mean cost throughout the season
df['player_season_roi'] = df['player_season_points'] / df['player_season_cost']
df = df[['player_name', 'team', 'position', 'player_season_points', 'player_season_bps', 'player_season_cost',  'player_season_minutes', 'player_season_roi']].drop_duplicates()
df['Season Points'] = df['player_season_points']
df['Position'] = df['position']
df_temp = df.sort_values(['player_season_points'],ascending=False).head(50)
fig = px.treemap(df_temp, path=[px.Constant('Position'), 'Position', 'player_name'], values='Season Points',
                 color='Season Points', 
                 color_continuous_scale=px.colors.sequential.algae)
fig.update_traces(root_color="lightgrey")
fig.update_layout(
    autosize=False,
    width=1000,
    height=500)
fig.show()
fig.write_image('C://Users//jd-vz//Desktop//Code//fig//treemap_top_players.pdf') # Update: Not working
# %%
# * Polar lot of positions and average points in teams (Complete)
df = pd.read_csv(data_file)
df['avg_position'] = df.groupby(['position', 'team'])['total_points'].transform('mean') # All player points in a season
df_temp = df[['avg_position', 'position', 'team']].drop_duplicates()
df_temp['Position'] = df_temp['position']
fig = px.bar_polar(df_temp, r="avg_position", theta="team", color="Position", template="plotly",
            color_discrete_sequence= px.colors.sequential.Plasma_r)
fig.write_image('C://Users//jd-vz//Desktop//Code//fig//polar_position_scores.pdf') #
fig.show()
# 1. Spurs (by far), 2. Leicester, and 3. Leeds have the best forwards
# 1. Leeds (tight), 2. Liverpool, 3. Burnely have the best goalies 
# 1. Man City (by far), Liverpool, Man Utd best midfielders
# 1. Liverpool, Man City, Chelsea have the best midfielders
# %%
# * Polar lot of premium players and average points in teams (Complete)
df = pd.read_csv(data_file)
df['premium_cutoff'] = df.groupby('position')['value'].transform('quantile', 0.75)
df['medium_cutoff'] = df.groupby('position')['value'].transform('quantile', 0.5)
df['premium_players'] = np.where(df['value'] >= df['premium_cutoff'], 'Premium', 
                                 np.where(df['value'] >= df['medium_cutoff'], 'Medium', 'Budget' ))
df['premium_position_pts'] = df.groupby(['team', 'premium_players'])['total_points'].transform('mean') # All player points in a season
df_temp = df[['premium_players', 'team', 'premium_position_pts']].drop_duplicates()
df_temp['Premium players'] = df['premium_players'] 
fig = px.bar_polar(df_temp, r="premium_position_pts", theta="team", color="Premium players", template="plotly",
            color_discrete_sequence= px.colors.sequential.Plasma_r, category_orders={'Premium players': ['Budget', 'Medium', 'Premium']})
fig.show()
fig.write_image('C://Users//jd-vz//Desktop//Code//fig//polar_premium_team_scores.pdf') #

# %%
# * Polar lot of amid vs dmid
df = pd.read_csv(data_file)
df = position_stats(df)
df = df[df['position_role'].isin(['AMID', 'DMID'])]
df['Midfielders'] = df['position_role']
df['premium_cutoff'] = df.groupby('Midfielders')['value'].transform('quantile', 0.75)
df['premium_players'] = np.where(df['value'] >= df['premium_cutoff'], 'Premium', 'Other') 
df['premium_position_pts'] = df.groupby(['team', 'premium_players', 'Midfielders'])['total_points'].transform('mean') # All player points in a season
df_temp = df[['premium_players', 'team', 'premium_position_pts', 'Midfielders']].drop_duplicates()
df_temp['Premium players'] = df['premium_players'] 
fig = px.bar_polar(df_temp, r="premium_position_pts", theta="team", color="Midfielders", template="plotly",
            color_discrete_sequence= px.colors.sequential.Plasma_r, category_orders={'Midfielders': ['AMID', 'DMID']})
fig.show()
fig.write_image('C://Users//jd-vz//Desktop//Code//fig//polar_amid_mid.pdf') #

# %%
# TODO: Winning
# ? Ratings percentage index = Wins/games played * 0.25 
df = pd.read_csv(data_file)
team_stats = df[['team', 'strength_attack_home', 'strength_attack_away',
                    'strength_defence_home', 'strength_defence_away', 'season']].drop_duplicates()
team_stats.rename(columns={'team': 'opponent_team', 'strength_attack_home': 'opponent_strength_attack_home',
                            'strength_attack_away': 'opponent_strength_attack_away', 
                            'strength_defence_home': 'opponent_strength_defence_home',
                            'strength_defence_away': 'opponent_strength_defence_away'}, inplace=True)
df = pd.merge(df, team_stats, on=['opponent_team', 'season'])
df['win'] = np.where(df['was_home'],
                     np.where(df['team_h_score'] > df['team_a_score'], 1, 0),
                     np.where(df['team_a_score'] > df['team_h_score'], 1, 0))
df['loss'] = np.where(df['was_home'],
                     np.where(df['team_h_score'] < df['team_a_score'], 1, 0),
                     np.where(df['team_a_score'] < df['team_h_score'], 1, 0))
df['draw'] = np.where(df['team_h_score'] == df['team_a_score'], 1, 0)
df = df.sort_values(['team', 'kickoff_time'])
df = df[['team', 'opponent_team', 'kickoff_time','win', 'loss', 'draw', 'season', 'GW']].drop_duplicates()
df['counter'] = 1
df['total_team_games'] = df.groupby(['team', 'season'])['counter'].cumsum() # Running team games
df['total_opponent_games'] = df.groupby(['opponent_team', 'season'])['counter'].cumsum() # Running opponent games
df['total_team_wins'] = df.groupby(['team', 'season'])['win'].cumsum() # Running team wins
df['total_team_losses'] = df.groupby(['team', 'season'])['loss'].cumsum() # Running team wins
df['total_team_draws'] = df.groupby(['team', 'season'])['draw'].cumsum() # Running team draws
df['total_opponent_team_wins'] =  df.groupby(['opponent_team', 'season'])['loss'].cumsum() # Running opponent wins
df['total_opponent_team_losses'] =  df.groupby(['opponent_team', 'season'])['win'].cumsum() # Running opponent loss
df['total_opponent_team_draws'] =  df.groupby(['opponent_team', 'season'])['draw'].cumsum() # Running opponent draws
df['TeamWinningProb'] = df['total_team_wins']/df['total_team_games']
df['TeamLosingProb'] = df['total_team_losses']/df['total_team_games']
df['TeamDrawingProb'] = df['total_team_draws']/df['total_team_games']
df['OppTeamWinningProb'] = df['total_opponent_team_wins']/df['total_opponent_games']
df['OppTeamLosingProb'] = df['total_opponent_team_losses']/df['total_opponent_games']
df['OppTeamDrawProb'] = df['total_opponent_team_draws']/df['total_opponent_games']
# df[['kickoff_time', 'team','total_team_games']].head(38)
df[['kickoff_time', 'team', 'total_team_games', 'win', 'loss', 'total_team_wins', 'total_team_losses', 
    'total_team_draws', 'TeamWinningProb', 'TeamLosingProb', 'TeamDrawingProb',
    'opponent_team', 'total_opponent_games', 'total_opponent_team_wins', 'total_opponent_team_losses',
    'OppTeamWinningProb', 'OppTeamLosingProb']].head(10)
df['TeamWinningProb'] + df['OppTeamLosingProb']

# %%
# * Is the team with the most wins the team with the most points 
df = pd.read_csv(data_file)
df['total_gw_pts'] = df.groupby(['team', 'season', 'GW'])['total_points'].transform('sum')
df = df[['team', 'opponent_team', 'was_home', 'team_h_score', 'team_a_score', 'season', 'total_gw_pts']].drop_duplicates()
df['win'] = np.where(df['was_home'],
                     np.where(df['team_h_score'] > df['team_a_score'], 1, 0),
                     np.where(df['team_a_score'] > df['team_h_score'], 1, 0))
df['total_team_wins'] = df.groupby(['team', 'season'])['win'].transform('sum')
df['TeamTotalPointsCount'] = df.groupby(['team', 'season'])['total_gw_pts'].transform('sum')
df = df[['team', 'TeamTotalPointsCount', 'total_team_wins']].drop_duplicates()
sns.set(rc={"figure.figsize":(12, 6)}) 
df.sort_values('TeamTotalPointsCount',ascending=False, inplace=True);
ax = sns.barplot(x="team", y="TeamTotalPointsCount", data=df, ci=None, label = 'Total team points scored', palette =sns.color_palette('Blues_r', 40))
plt.xticks(rotation=90)
plt.xlabel('Teams')
plt.ylabel('Team points')
plt.grid(False)
width_scale = 0.5
for bar in ax.containers[0]:
    bar.set_width(bar.get_width() * width_scale)
ax2 = ax.twinx()
ax_3 = sns.barplot(x='team', y='total_team_wins', label = 'Total team wins', data=df, ci=None,ax=ax2, color = 'gray')
plt.ylabel('Wins')
# ax_3.legend(loc='upper right')
for bar in ax2.containers[0]:
    x = bar.get_x()
    w = bar.get_width()
    bar.set_x(x + w * (1- width_scale))
    bar.set_width(w * width_scale)
plt.grid(False)
handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax_3.get_legend_handles_labels())]
ax_3.legend(handles, labels, loc='upper right')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//team_wins_to_pts.pdf', bbox_inches='tight') 

# %%
# Plot top fifty players cumulative minutes vs their points scored
import matplotlib
from matplotlib.lines import Line2D
df = pd.read_csv(data_file)
plt.figure(figsize=(15, 8))
df['player_season_points'] = df.groupby(['player_name', 'season'])['total_points'].transform('sum') # All player points in a season
df['player_season_minutes'] = df.groupby(['player_name', 'season'])['minutes'].transform('sum') # All minutes in season
df = df[['player_name', 'player_season_points', 'player_season_minutes', 'position', 'season']].drop_duplicates()

df = df.sort_values(['player_season_points'],ascending=False)
df_1 = df[df['position'] == 'FWD'].head(15)
df_2 = df[df['position'] == 'MID'].head(15)
df_3 = df[df['position'] == 'GK'].head(15)
df_4 = df[df['position'] == 'DEF'].head(15)

df_temp = pd.concat([df_3, df_4, df_2, df_1])
palette = [color for color in [[matplotlib.colors.to_hex(sns.color_palette('YlOrBr_r', 20)[i]) for i in range(15)],
                               [matplotlib.colors.to_hex(sns.color_palette('YlGn_r', 20)[i]) for i in range(15)],
                               [matplotlib.colors.to_hex(sns.color_palette('PuBu_r', 20)[i]) for i in range(15)],
                               [matplotlib.colors.to_hex(sns.color_palette('PiYG', 40)[i]) for i in range(15)]]] 

flt = list(np.array(palette).ravel())
ax = sns.barplot(x='player_name', y='player_season_points', label = 'Points scored', data = df_temp, ci=None, palette=flt)
plt.xticks(rotation=90)
plt.xlabel('Player name')
plt.ylabel('Points scored') 
plt.grid(False)
colors = [sns.color_palette('YlOrBr_r', 20)[0],
          sns.color_palette('YlGn_r', 20)[0],
          sns.color_palette('PuBu_r', 20)[0],
          sns.color_palette('PiYG', 40)[0]]

legend_elements = [Line2D([0], [0], color=colors[0], lw=3, label='GK'),
                   Line2D([0], [0], color=colors[1], lw=3, label='DEF'),
                   Line2D([0], [0], color=colors[2], lw=3, label='MID'), 
                   Line2D([0], [0], color=colors[3], lw=3, label='FWD')]

ax.legend(handles=legend_elements, loc='upper right')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//player_season_win_pts.pdf', bbox_inches='tight') 
# %%
# * Plot minutes points to average match points
df = pd.read_csv(data_file)
plt.figure(figsize=(12, 6))
BIN_VALUE = 4
df['minutes_bins'] = pd.cut(df['minutes'], bins=BIN_VALUE,  labels = ['[0, 22.5]', '(22.5, 45]', '(45, 67.5]', '(67.5,90]'])
ax = sns.catplot(data=df, x="minutes_bins", y = 'total_points', col="position", palette = 'inferno', kind = 'bar', ci = None)
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//player_game_min_pts.pdf', bbox_inches='tight') 
# %%
# * Plot cumulative season points to cumulative season minutes
df = pd.read_csv(data_file)
df['Cumulative season points'] = df.groupby(['player_name', 'season'])['total_points'].transform('sum') # All player points in a season
df['Cumulative season minutes'] = df.groupby(['player_name', 'season'])['minutes'].transform('sum') # All minutes played throughout the season
df['Position'] = df['position']
sns.scatterplot(x = df['Cumulative season minutes'], y = df['Cumulative season points'], hue = df['Position'])
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//player_game_min_pts.png', bbox_inches='tight') 
#


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
sns.set(rc={"figure.figsize":(12, 6)}) 
training_path = 'C://Users//jd-vz//Desktop//Code//data//'
df = pd.read_csv(training_path + 'collected_fpl.csv')
df['Season'] = df['season']
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows =2, sharey=False, sharex = False)
sns.histplot(x = 'minutes', data = df, fill = True, edgecolor = 'black', hue = 'Season', palette = 'pastel', alpha = 0.9, multiple = 'stack', ax = ax1, legend = False, bins = 10).set(xlabel=None)
sns.histplot(x = 'total_points', data = df, fill = True, edgecolor = 'black', hue = 'Season', palette = 'pastel', alpha = 0.9, multiple = 'stack', bins = 20, legend = True, ax = ax2).set(xlabel=None)
df = pd.read_csv(training_path + 'collected_us.csv')
df['Season'] = df['season']
sns.histplot(x = 'minutes', data = df, fill = True, edgecolor = 'black', hue = 'Season', palette = 'pastel', alpha = 0.9, multiple = 'stack', ax = ax3, legend = False, bins = 10).set(xlabel=None)
sns.histplot(x = 'total_points', data = df, fill = True, edgecolor = 'black', hue = 'Season', palette = 'pastel', alpha = 0.9, multiple = 'stack', legend = False, bins = 20, ax = ax4).set(xlabel=None)
ax1.title.set_text('Minutes')
ax1.set_ylabel('Fantasy')
ax2.set_ylabel('')
ax3.set_ylabel('Understat')
ax4.set_ylabel('')
ax2.title.set_text('Total points')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//fpl_us_target_dist.pdf', bbox_inches='tight') 

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
sns.set(rc={"figure.figsize":(12, 6)}) 
training_path = 'C://Users//jd-vz//Desktop//Code//data//'
df = pd.read_csv(training_path + 'collected_us.csv')
df['Season'] = df['season']
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
sns.histplot(x = 'minutes', data = df, fill = True, hue = 'Season', palette = 'pastel', alpha = 0.9, multiple = 'stack', ax = ax1, legend = False, bins = 10)
sns.histplot(x = 'total_points', data = df, fill = True, hue = 'Season', palette = 'pastel', alpha = 0.9, multiple = 'stack', bins = 20, ax = ax2)
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//us_target_dist.pdf', bbox_inches='tight') 
# %%
# Over aachiever
df = pd.read_csv(data_file)
df['premium_cutoff'] = df.groupby(['GW', 'season', 'position'])['value'].transform('quantile', 0.75)
df['medium_cutoff'] = df.groupby(['GW', 'season', 'position'])['value'].transform('quantile', 0.5)
df['premium_players'] = np.where(df['value'] >= df['premium_cutoff'], 'Premium', np.where(df['value'] >= df['medium_cutoff'], 'Medium', 'Budget'))
df['over_achiever_cutoff'] =  df.groupby(['GW', 'season', 'position'])['total_points'].transform('quantile', 0.70)
df['over_achiever'] = np.where(df['premium_players'].isin(['Medium', 'Budget']) & (df['total_points'] > df['over_achiever_cutoff']),  True, False)
# %%
df.iloc[df['transfers_balance'].idxmin()]
# %%
# * 4e) The number of unique players used in fixtures 

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
df = pd.read_csv(data_file)
df['Season'] = df['season']
df_1 = df[['team', 'opponent_team','kickoff_time', 'GW', 'Season']].drop_duplicates()
df_2 = df_1.groupby(['GW', 'Season']).size().reset_index(name='Num_Matches_GW') 
df_3 = df[['player_name', 'GW', 'Season']].drop_duplicates()
df_4 = df.groupby(['GW', 'Season']).size().reset_index(name='Num_Players_GW') 
sns.histplot(data = df_4, x='GW', hue='Season', weights='Num_Players_GW', multiple='stack',bins=38, palette='pastel', edgecolor = 'black', ax = ax2, legend = False)
ax1.set_ylabel('Fixtures')
ax2.set_ylabel('Players')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//players_gameweeks.pdf', bbox_inches='tight') 
# %%
def create_premium_players(df): 
    df['premium_cutoff'] = df.groupby(['GW', 'season', 'position'])['value'].transform('quantile', 0.75)
    df['medium_cutoff'] = df.groupby(['GW', 'season', 'position'])['value'].transform('quantile', 0.5)
    df['premium_players'] = np.where(df['value'] >= df['premium_cutoff'], 'Premium', 'Budget')
    df['over_achiever_cutoff'] =  df.groupby(['GW', 'season', 'position'])['total_points'].transform('quantile', 0.7)
    df['over_achiever_val_cutoff'] =  df.groupby(['GW', 'season', 'position'])['value'].transform('quantile', 0.7)
    df['over_achiever'] = np.where((df['value'] < df['over_achiever_val_cutoff'] ) & (df['total_points'] > df['over_achiever_cutoff']),  True, False)
    df = df.drop(['over_achiever_cutoff', 'medium_cutoff', 'premium_cutoff'], axis = 1)
    return df

df = pd.read_csv(data_file)
df = create_premium_players(df)
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
df['FDR'] = np.where(df['opponent_team_overall'] > df['player_team_overall'] + 10, 'High',
                     np.where(df['opponent_team_overall'] < df['player_team_overall'] - 10, 'Low', 'Med')) 
df['Season'] = df['season']
df['Cost bracket'] = df['premium_players']
df['Total points'] = df['total_points']
df['Position'] = df['position']
# df = df[df['premium_players'].isin(['Medium', 'Budget'])]
g = sns.FacetGrid(df, col="Position", col_order = ['GK', 'DEF', 'MID', 'FWD'],legend_out =True)
g.map(sns.pointplot, "Cost bracket", "Total points", 'over_achiever', 
      ci=None, order = ['Budget', 'Premium'], hue_order = [False, True], palette = 'deep' )
# %%
ax = sns.scatterplot(x = 'value', y = 'total_points', hue = 'position', style = 'over_achiever',
                data = df,  edgecolor="gray", alpha = 0.8)
# %%
# * 4e) The number of unique players used in fixtures 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
df = pd.read_csv(data_file)
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows =1, sharey=False, sharex = False, figsize = (20, 5))
df['Season'] = df['season']
df_1 = df[['team', 'opponent_team','kickoff_time', 'GW', 'Season']].drop_duplicates()
df_2 = df_1.groupby(['GW', 'Season']).size().reset_index(name='Num_Matches_GW') 
df_3 = df[['player_name', 'GW', 'Season']].drop_duplicates()
df_4 = df.groupby(['GW', 'Season']).size().reset_index(name='Num_Players_GW') 
sns.histplot(data = df_2, x='GW', hue='Season', weights='Num_Matches_GW', multiple='stack',bins=38, palette='pastel', edgecolor = 'black', ax = ax1, legend = True)
sns.histplot(data = df_4, x='GW', hue='Season', weights='Num_Players_GW', multiple='stack',bins=38, palette='pastel', edgecolor = 'black', ax = ax2, legend = False)
ax1.set_ylabel('Fixtures')
ax2.set_ylabel('Players')
ax1.set_xlabel('Gameweek')
ax2.set_xlabel('Gameweek')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//players_gameweeks.pdf', bbox_inches='tight') 
# %%
# * 4e) The minimum number of unique players used in fixtures 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
df = pd.read_csv(data_file)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows =2, sharey=False, sharex = False, figsize = (20, 5))
df['Season'] = df['season']
df_1 = df[['team', 'opponent_team','kickoff_time', 'GW', 'Season']].drop_duplicates()
df_2 = df_1.groupby(['GW', 'Season'])['position'].reset_index(name='Num_Matches_GW') 
sns.histplot(data = df_2, x='GW', hue='Season', weights='Num_Matches_GW', multiple='stack',bins=38, palette='pastel', edgecolor = 'black', ax = ax1, legend = True)
sns.histplot(data = df_4, x='GW', hue='Season', weights='Num_Players_GW', multiple='stack',bins=38, palette='pastel', edgecolor = 'black', ax = ax2, legend = False)
ax1.set_ylabel('Fixtures')
ax2.set_ylabel('Players')
ax1.set_xlabel('Gameweek')
ax2.set_xlabel('Gameweek')

df = pd.read_csv(data_file)
df = df.groupby(['position', 'team', 'kickoff_time', 'season']).size().reset_index(name='counts') 
df['total_position_counts'] = df.groupby(['position', 'team', 'kickoff_time'])['counts'].transform('sum')
df.sort_values(['total_position_counts'], ascending=False, inplace=True)
df['Positions'] = df['position']
df['Season'] = df['season']
sns.boxenplot(x = 'Positions', y = 'total_position_counts', order= ['DEF', 'MID', 'GK', 'FWD'],
                data = df, ax = ax3,
                 palette = 'pastel')
df = pd.read_csv(data_file) 
df['Positions'] = df['position']
sns.kdeplot(x = 'total_points', hue = 'Positions', data = df,
            multiple="stack", palette='pastel', edgecolor="black", ax = ax4, alpha = 0.7)
plt.xlabel('Total points')
ax1.set_ylabel('Counts')
ax2.set_ylabel('Density')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//players_gameweeks.pdf', bbox_inches='tight') 
# %%
df = pd.read_csv(data_file)
fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows =1, sharey=False, sharex = False, figsize = (20,5))
df['counts'] = df.groupby(['position', 'team', 'kickoff_time', 'season']).size().reset_index(name='counts') 



df['total_position_counts'] = df.groupby(['position', 'team', 'kickoff_time'])['counts'].transform('sum')
df.sort_values(['total_position_counts'], ascending=False, inplace=True)
df['Positions'] = df['position']
df['Season'] = df['season']
sns.boxenplot(x = 'Positions', y = 'total_position_counts', order= ['DEF', 'MID', 'GK', 'FWD'],
                data = df, ax = ax1,
                 palette = 'pastel')
df = pd.read_csv(data_file) 
df['Positions'] = df['position']
sns.kdeplot(x = 'total_points', hue = 'Positions', data = df,
            multiple="stack", palette='pastel', edgecolor="black", ax = ax2, alpha = 0.7)
plt.xlabel('Total points')
ax1.set_ylabel('Counts')
ax2.set_ylabel('Density')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//positions_p_occurences.pdf', bbox_inches='tight') 

# %%
df = pd.read_csv(data_file)
fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows =1, sharey=False, sharex = False, figsize = (20,5))
df = df.groupby(['position', 'team', 'kickoff_time', 'season']).size().reset_index(name='counts') 
df['total_position_counts'] = df.groupby(['position', 'team', 'kickoff_time'])['counts'].transform('sum')
df.sort_values(['total_position_counts'], ascending=False, inplace=True)
df['Positions'] = df['position']
df['Season'] = df['season']
sns.boxenplot(x = 'Positions', y = 'total_position_counts', order= ['DEF', 'MID', 'GK', 'FWD'],
                data = df, ax = ax1,                  palette = 'pastel')
df = pd.read_csv(data_file) 
df['Positions'] = df['position']
df['total_position_minutes'] = df.groupby(['position', 'team', 'kickoff_time'])['minutes'].transform('sum')
sns.histplot(x = 'minutes', hue = 'Positions', data = df, edgecolor = 'black',
           palette='pastel', fill = True, multiple= 'stack',ax = ax2, cumulative = True)
plt.xlabel('Minutes')
ax1.set_ylabel('Count')
ax2.set_ylabel('Count')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//positions_p_occurences.pdf', bbox_inches='tight') 
# %%
# Avg minutes per team
df = pd.read_csv(data_file)
fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows =1, sharey=False, sharex = False, figsize = (20,5))
# df = df.groupby(['position', 'team', 'kickoff_time', 'season']).size().reset_index(name='counts') 
df['avg_position_minutes'] = df.groupby(['position', 'team', 'kickoff_time'])['minutes'].transform('mean')
# df.sort_values(['total_position_counts'], ascending=False, inplace=True)
df['Positions'] = df['position']
df['Season'] = df['season']
sns.boxenplot(x = 'Positions', y = 'avg_position_minutes', order= ['DEF', 'MID', 'GK', 'FWD'],
                data = df, ax = ax1, palette = 'pastel')
df = pd.read_csv(data_file) 
df['total_team_minutes'] = df.groupby(['team', 'GW', 'kickoff_time'])['minutes'].transform('sum') / 11
df = df[['team', 'GW', 'total_team_minutes', 'season']].drop_duplicates()
sns.barplot(x = 'GW', y = 'total_team_minutes', hue = 'season', dodge = True, data = df, edgecolor = 'black', palette='pastel', ax = ax2, ci = None)
plt.xlabel('Gameweek')
plt.xticks(rotation=90)
ax1.set_ylabel('Minutes')
ax2.set_ylabel('Minutes (Average)')
ax2.legend_.remove()
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//teams_p_occurences.pdf', bbox_inches='tight') 
# %%
# Avg minutes per team
df = pd.read_csv(data_file)
fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows =1, sharey=False, sharex = False, figsize = (20,5))
df['counter'] = 1
df['player_occurrence_in_team'] = df.groupby(['team', 'kickoff_time'])['counter'].transform('sum')
df['Positions'] = df['position']
df['Season'] = df['season']
sns.kdeplot(x = 'player_occurrence_in_team', hue = 'Positions', fill = True, multiple = 'stack', edgecolor = 'black',
                data = df, ax = ax1, palette = 'pastel')
plt.xlabel('Players in team')

# %%
df = pd.read_csv(data_file) 
df['total_team_minutes'] = df.groupby(['team', 'GW', 'kickoff_time'])['minutes'].transform('sum') / 11
df = df[['team', 'GW', 'total_team_minutes', 'season']].drop_duplicates()
sns.barplot(x = 'GW', y = 'total_team_minutes', hue = 'season', dodge = True, data = df, edgecolor = 'black', palette='pastel', ax = ax2, ci = None)
plt.xlabel('Gameweek')
plt.xticks(rotation=90)
ax1.set_ylabel('Minutes')
ax2.set_ylabel('Minutes (Average)')
ax2.legend_.remove()
# plt.savefig('C://Users//jd-vz//Desktop//Code//fig//teams_p_occurences.pdf', bbox_inches='tight') 
# %%
# Avg minutes per team
df = pd.read_csv(data_file)
fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows =1, sharey=False, sharex = False, figsize = (20,5))
# df = df.groupby(['position', 'team', 'kickoff_time', 'season']).size().reset_index(name='counts') 
df['avg_position_minutes'] = df.groupby(['position', 'team', 'kickoff_time'])['minutes'].transform('mean')
# df.sort_values(['total_position_counts'], ascending=False, inplace=True)
df['Positions'] = df['position']
df['Season'] = df['season']
sns.boxenplot(x = 'Positions', y = 'avg_position_minutes', order= ['DEF', 'MID', 'GK', 'FWD'],
                data = df, ax = ax1, palette = 'pastel')
df['counter'] = 1
df['player_occurrence_in_team'] = df.groupby(['team', 'kickoff_time'])['counter'].transform('sum')
df['Positions'] = df['position']
df['Season'] = df['season']
sns.histplot(x = 'player_occurrence_in_team', hue = 'Positions', fill = True, multiple = 'stack', edgecolor = 'black', bins = 7,
                data = df, ax = ax2, palette = 'pastel')
ax1.set_ylabel('Minutes')
ax2.set_ylabel('Count')
ax2.set_xlabel('Player measurements in fixtures')
plt.savefig('C://Users//jd-vz//Desktop//Code//fig//positions_times_occurences.pdf', bbox_inches='tight') 
# %%
