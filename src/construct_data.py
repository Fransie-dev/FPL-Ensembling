# %%
import pandas as pd
import matplotlib.pyplot as plt

def read_data(season):
    """[This function reads the data and defines the features that will be averaged]

    Returns:
        [type]: [description]
    """    
    understat = pd.read_csv(f'C://Users//jd-vz//Desktop//Code//data//{season}//training//cleaned_understat.csv', index_col=0)
    fpl = pd.read_csv(f'C://Users//jd-vz//Desktop//Code//data//{season}//training//cleaned_fpl.csv', index_col=0)
    fpl_avg = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored', 
               'ict_index', 'influence', 'saves', 'threat']
    understat_avg = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored', 
                     'ict_index', 'influence', 'saves', 'threat',
                     'shots', 'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup']
    fpl_lag = fpl_avg + ['total_points']
    understat_lag = understat_avg + ['total_points']
    return fpl, fpl_avg, understat, understat_avg

def rolling_avg(data, prev_games, feats):
    """[This function creates a previous game rolling average for the selected features]

    Args:
        data ([type]): [description]
        prev_games ([type]): [description]
        feats ([type]): [description]

    Returns:
        [type]: [description]
    """    
    new_feats = []
    for feat in feats:
        new_feats.append(feat + '_last_' + str(prev_games))
    data[new_feats] = data[feats].rolling(min_periods=1, window=prev_games).mean()
    return data

def plot_player(player, feature, prev_games, data):
    subset = data.loc[data.player_name == player]
    plt.plot(subset['GW'], subset[feature], label = feature)
    plt.plot(subset['GW'], subset[feature + '_last_' + str(prev_games)], label = feature +  '_last_' + str(prev_games))
    plt.legend()
    plt.show()

def plot_Ozil(understat_avg, prev_games, understat):
    for feat in understat.columns:
        plot_player('Mesut Özil', feat, prev_games=prev_games, data = understat)

def main(season, prev_games):
    fpl, fpl_avg, understat, understat_avg = read_data(season)
    fpl = fpl.groupby(['player_name']).apply(rolling_avg, prev_games = prev_games, feats=fpl_avg)
    understat = understat.groupby(['player_name']).apply(rolling_avg, prev_games = prev_games, feats=understat_avg)
    plot_Ozil(understat_avg, prev_games, understat)
    
# %%
if __name__ == '__main__':
    main('2019-20', prev_games=12)
    
    
# %%
# %%
