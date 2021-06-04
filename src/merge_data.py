# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Imports and functions 
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
import pandas as pd
import numpy as np
import difflib


def read_data(fpl_path, understat_path):
    """[This function reads in the data from collect_data.py and returns the4 unmerged understat and fpl data]

    Args:
        fpl_path ([type]): [description]
        understat_path ([type]): [description]

    Returns:
        [understat]: ['goals','shots','xG','time','position','h_team','a_team','h_goals','a_goals','date','id','season','roster_id','xA','assists','key_passes','npg','npxG' 'xGChain' 'xGBuildup', 'player_name']
        [fpl_to_understat]: ['understat_team', 'understat_pos', 'name', 'FPL_ID', 'Understat_ID']
        [fpl]: [FPL api data]

    """
    understat = pd.read_csv(understat_path + 'all_understat_players.csv')
    understat['date']=pd.to_datetime(understat['date']).dt.date
    
    
    fpl_to_understat = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//FPL_Understat_IDs.csv')
    fpl_to_understat['name'] = fpl_to_understat['fpl_first_name'] + ' ' + fpl_to_understat['fpl_last_name']
    fpl_to_understat = fpl_to_understat[['team', 'pos', 'name', 'FPL_ID', 'Understat_ID']]
    fpl_to_understat.rename(columns = {'team':'understat_team', 'pos':'understat_pos', 'name':'player_name'},inplace = True) 
    
        
    fpl = pd.read_csv(fpl_path + 'merged_gw.csv') 
    fpl.rename(columns = {'element':'FPL_ID', 'name':'player_name'},inplace = True) 
    fpl['kickoff_time'] = pd.to_datetime(fpl['kickoff_time']).dt.date

    
    fixtures = pd.read_csv(data_path + 'fixtures.csv')
    fixtures = fixtures[['id', 'team_a', 'team_a_difficulty', 'team_h', 'team_h_difficulty']]
    fixtures.rename(columns={'id':'fixture'}, inplace=True)

    
    teams = pd.read_csv(data_path + 'teams.csv')
    teams = teams[['id', 'name', 'short_name', 'strength', 'strength_attack_away', 'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 'strength_overall_home']]

    
    return understat, fpl_to_understat, fpl, teams, fixtures

def rename_teams(teams,fpl, cols = ['team','opponent_team', 'team_a', 'team_h']):
    """[This function maps the team id to the fpl dataset]

    Args:
        cols (list, optional): [description]. Defaults to ['opponent_team', 'team_a', 'team_h'].
    """    
    for col in cols:
        for i in np.arange(start = 0, stop = len(teams)):
            id = teams['id'][i]
            team_name = teams['name'][i]
            fpl[col].loc[fpl[col] == id] = team_name
    return fpl

def intersect(a, b):
    print(len(list(set(a) & set(b))), 'unique, matching entries within the two columns')
    return list(set(a) & set(b))

def union(a, b):
    print(len(list(set(a) | set(b))), 'unique, not necessarily matching entries within the two columns')
    return list(set(a) | set(b))
    

def merge_fpl_data(fpl, fixtures, teams):
    fpl = pd.merge(fpl, fixtures, on='fixture') # Now have some match statistics
    fpl = pd.merge(fpl, teams, left_on='team', right_on='name')
    fpl = rename_teams(teams, fpl)
    return fpl
    
def rename_fpl_teams(fpl):
    team_reps = {
        'Man City':'Manchester City', 
        'Man Utd': 'Manchester United', 
        'West Brom':'West Bromwich Albion',
        'Spurs':'Tottenham',
        'Sheffield Utd':'Sheffield United',
        'West Ham':'West Ham',
        'Wolves':'Wolverhampton Wanderers',
        'Brighton':'Brighton',
        'Chelsea':'Chelsea',
        'Newcastle':'Newcastle',
        'Everton':'Everton',
        'Fulham':'Fulham',
        'Arsenal':'Arsenal',
        'Leeds':'Leeds',
        'Liverpool':'Liverpool',
        'Leicester':'Leicester',
        'Southampton':'Southampton',
        'Crystal Palace':'Crystal Palace',
        'Aston Villa':'Aston Villa',
        'Burnley':'Burnley',
        'Watford':'Watford',
        'Bournemouth':'Bournemouth', 
        'Norwich':'Norwich'
        }
    fpl['team'] = fpl['team'].map(team_reps)
    fpl['team_a'] = fpl['team_a'].map(team_reps)
    fpl['team_h'] = fpl['team_h'].map(team_reps)
    return fpl


def get_matching_names(understat_names, fpl_names, threshold):
    seq = difflib.SequenceMatcher()
    understat_similar = []
    fpl_similar = []
    ratio = []
    for i in range(len(understat_names)):
        for j in range(len(fpl_names)):
            seq.set_seqs(understat_names[i].lower(), fpl_names[j].lower())
            ratio_similar = seq.ratio()
            if ratio_similar > threshold:
                understat_similar.append(understat_names[i])
                fpl_similar.append(fpl_names[j])
                ratio.append(ratio_similar)
    similarity_matched_df = pd.DataFrame({'understat':understat_similar, 'fpl':fpl_similar, 'similarity': ratio})
    similarity_matched_df_final = similarity_matched_df.loc[similarity_matched_df.groupby('understat')['similarity'].idxmax()]
    return similarity_matched_df_final


    
    
# season = '2019-20' 
season = '2020-21'
fpl_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//gws//' 
understat_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//understat//'
data_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//'
understat, fpl_to_understat, fpl, teams, fixtures = read_data(fpl_path, understat_path)
fpl = merge_fpl_data(fpl, fixtures, teams) # No understat statistics included in this dataframe
fpl = rename_fpl_teams(fpl)


# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# First match the entries whos names and dates match exactly
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
matching_names = intersect(understat['player_name'],fpl['player_name'])
fpl_matched = fpl[fpl['player_name'].isin(matching_names)]
understat_matched = understat[understat['player_name'].isin(matching_names)]
names_merged_on_date = pd.merge(fpl_matched, understat_matched, left_on=['player_name', 'kickoff_time'], right_on=['player_name', 'date']) 
# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Now find the entries whos names and dates DID NOT match exactly and use a similarity finder to
# manually look for potential spelling errors
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
fpl_not_matched = fpl[~fpl['player_name'].isin(names_merged_on_date['player_name'].unique())] 
understat_not_matched = understat[~understat['player_name'].isin(names_merged_on_date['player_name'].unique())]
similarity_matched_df = get_matching_names(understat_not_matched['player_name'].unique(),
                                           fpl_not_matched['player_name'].unique(),
                                           threshold=0.5) # Returns a dataframe of matching names with a degree of similarity
wrongly_matched_names = ['Alisson', 'Allan', 'Ederson', 'Emerson',  'Joelinton', 
                         'André Gomes', 'Ben Chilwell', 'Bernardo Silva', 'David Luiz', 
                         'Felipe Anderson', 'Ricardo Pereira', 'Lucas Moura', 'Rodri', 'Rúben Dias',
                         'Rúben Vinagre', 'Trézéguet', 'Vitinha', 'Willian'] # Manually look for names that do not logically make sense in similarity_matched_df
wrongly_matched_entries = understat_not_matched[~understat_not_matched['player_name'].isin(understat_not_matched)]
similarity_matched_df = similarity_matched_df[~similarity_matched_df['understat'].isin(wrongly_matched_names)] # Remove all those names that have no logical match
name_mapper = dict(zip(similarity_matched_df['understat'], similarity_matched_df['fpl'])) # Create dictionary to rename
updt = dict(zip(wrongly_matched_names, wrongly_matched_names)) # Create dictionary to keep wrongly mapped names
name_mapper.update(updt)
name_mapper
# TODO: SEE WHY UNDERSTAT_NOT_MATCHED is throwing NA's
# %%
print(understat_not_matched['player_name'].unique())
understat_not_matched['player_name'] = understat_not_matched['player_name'].map(name_mapper)  # Rename, note all the wrongly matched ones are removed from the dataset
print(understat_not_matched['player_name'].unique())























# %%
names_merged_on_similarity = pd.merge(fpl_not_matched, understat_not_matched, left_on=['player_name', 'kickoff_time'],
                        right_on=['player_name', 'date']) # Merge using player name and date of game
# %%
# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Now find the entries whos names and dates still did not match exactly by lowering the threshold 
# and removing them from the pool
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
fpl_still_not_matched = fpl_not_matched[~fpl_not_matched['player_name'].isin(union(names_merged_on_similarity['player_name'].unique(),
                                                                                   names_merged_on_date['player_name'].unique()))] # Names that were not within the similarity dataframe
understat_still_not_matched = understat_not_matched[~understat_not_matched['player_name'].isin(union(names_merged_on_similarity['player_name'].unique(),
                                                                                   names_merged_on_date['player_name'].unique()))] # Names that were not within the similarity dataframe
similarity_matched_df = get_matching_names(understat_still_not_matched['player_name'].unique(),
                                           fpl_still_not_matched['player_name'].unique(),
                                           threshold=0.5) # Returns a dataframe of matching names with a degree of similarity
similarity_matched_df
# %%
names_merged_on_game = pd.merge(fpl_still_not_matched, understat_still_not_matched, left_on=['player_name', 'team_a', 'team_h'], right_on=['player_name','a_team', 'h_team'])


# %%
union(names_merged_on_similarity['player_name'].unique(),names_merged_on_date['player_name'].unique())

# %%
