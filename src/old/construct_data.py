# %%
import pandas as pd
import matplotlib.pyplot as plt
from utilities import delete_any_duplicates

def read_data(season = '2019-00'):
    """[This function reads the data and defines the features that will be averaged]

    Returns:
        [type]: [description]
    """    
    understat = delete_any_duplicates(pd.read_csv(f'C://Users//jd-vz//Desktop//Code//data//{season}//training//cleaned_understat.csv', index_col=0))
    fpl = delete_any_duplicates(pd.read_csv(f'C://Users//jd-vz//Desktop//Code//data//{season}//training//cleaned_fpl.csv', index_col=0))
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
    # plt.plot(subset['GW'], subset[feature + '_last_' + str(prev_games)], label = feature +  '_last_' + str(prev_games))
    plt.legend()
    plt.show()

def plot_Ozil(understat_avg, prev_games, understat):
    for feat in understat_avg:
        plot_player('Mesut Özil', feat, prev_games=prev_games, data = understat)

def main(season, prev_games):
    fpl, fpl_avg, understat, understat_avg = read_data(season)
    fpl = fpl.groupby(['player_name']).apply(rolling_avg, prev_games = prev_games, feats=fpl_avg)
    understat = understat.groupby(['player_name']).apply(rolling_avg, prev_games = prev_games, feats=understat_avg)
    # plot_Ozil(understat_avg, prev_games, understat)
    
main(season='2019-20', prev_games = 1)
    
# %%
if __name__ == '__main__':
    fpl, fpl_avg, understat, understat_avg = read_data(season='2019-20')
    plot_Ozil(understat_avg=fpl.columns.drop(['GW', 'player_name','home_True', 'position_DEF', 'position_FWD',
                                                'position_GK', 'position_MID', 
                                                'kickoff_time',
                                                'team_h_Arsenal', 'team_h_Aston Villa',
                                                'team_h_Bournemouth', 'team_h_Brighton', 'team_h_Burnley',
                                                'team_h_Chelsea', 'team_h_Crystal Palace', 'team_h_Everton',
                                                'team_h_Leicester', 'team_h_Liverpool', 'team_h_Manchester City',
                                                'team_h_Manchester United', 'team_h_Newcastle', 'team_h_Norwich',
                                                'team_h_Sheffield United', 'team_h_Southampton', 'team_h_Tottenham',
                                                'team_h_Watford', 'team_h_West Ham', 'team_h_Wolverhampton Wanderers',
                                                'team_a_Arsenal', 'team_a_Aston Villa', 'team_a_Bournemouth',
                                                'team_a_Brighton', 'team_a_Burnley', 'team_a_Chelsea',
                                                'team_a_Crystal Palace', 'team_a_Everton', 'team_a_Leicester',
                                                'team_a_Liverpool', 'team_a_Manchester City',
                                                'team_a_Manchester United', 'team_a_Newcastle', 'team_a_Norwich',
                                                'team_a_Sheffield United', 'team_a_Southampton', 'team_a_Tottenham',
                                                'team_a_Watford', 'team_a_West Ham', 'team_a_Wolverhampton Wanderers']), prev_games=3, understat = understat)
    
    
    
# %%
# %%

# %%
