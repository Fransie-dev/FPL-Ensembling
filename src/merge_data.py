# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Imports and functions 
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
import pandas as pd
# pd.set_option('display.max_rows', None) # Slows down code too much, just view from variable explorer
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
    """[This function replaces the integer value used to indicate the teams with the FPL's associated string]

    Args:
        teams ([type]): [description]
        fpl ([type]): [description]
        cols (list, optional): [description]. Defaults to ['team','opponent_team', 'team_a', 'team_h'].

    Returns:
        [type]: [A converted dataframe]
    """
    for col in cols:
        for i in np.arange(start = 0, stop = len(teams)):
            id = teams['id'][i]
            team_name = teams['name'][i]
            fpl[col].loc[fpl[col] == id] = team_name
    return fpl

def intersect(a, b):
    """[This function finds the intersection between two dataframe columns]

    Args:
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [The intersection]
    """    
    print(len(list(set(a) & set(b))), 'unique and matching names between FPL and Understat')
    return list(set(a) & set(b))

def union(a, b):
    """[This function finds the union between two dataframe columns]

    Args:
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [The union]
    """    
    print(len(list(set(a) | set(b))), 'unique, not necessarily matching entries within the two columns')
    return list(set(a) | set(b))
    

def merge_fpl_data(fpl, fixtures, teams):
    """[This function merges the relevant data scraped from the FPL API]

    Args:
        fpl ([type]): [description]
        fixtures ([type]): [description]
        teams ([type]): [description]

    Returns:
        [type]: [A merged dataframe]
    """    
    fpl = pd.merge(fpl, fixtures, on='fixture') # Now have some match statistics
    fpl = pd.merge(fpl, teams, left_on='team', right_on='name')
    fpl = rename_teams(teams, fpl)
    return fpl
    
def rename_fpl_teams(fpl):
    """[This function replaces all the acronyms used by the FPL API for the different teams]

    Args:
        fpl ([type]): [description]

    Returns:
        [type]: [A renamed dataframe]
    """    
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


def get_matching_names(understat_names, fpl_names):
    """[This function checks for similarity between understat and fpl names, and renames the most matching understat
    name to the associated FPL name.]

    Args:
        understat_names ([type]): [description]
        fpl_names ([type]): [description]
        threshold ([type]): [description]

    Returns:
        [type]: [description]
    """    
    seq = difflib.SequenceMatcher()
    understat_similar = []
    fpl_similar = []
    ratio = []
    for i in range(len(understat_names)):
        for j in range(len(fpl_names)):
            seq.set_seqs(understat_names[i].lower(), fpl_names[j].lower())
            ratio_similar = seq.ratio()
            understat_similar.append(understat_names[i])
            fpl_similar.append(fpl_names[j])
            ratio.append(ratio_similar)
    similarity_matched_df = pd.DataFrame({'understat':understat_similar, 'fpl':fpl_similar, 'similarity': ratio})
    similarity_matched_df_final = similarity_matched_df.loc[similarity_matched_df.groupby('understat')['similarity'].idxmax()]
    print(similarity_matched_df_final)
    return similarity_matched_df_final

def exact_matches(understat, fpl):
    """[This function matches the entries whos names and dates match exactly]

    Args:
        understat ([type]): [description]
        fpl ([type]): [description]

    Returns:
        [type]: [The merged data, and the sets of data used to construct it]
    """    
    matching_names = intersect(understat['player_name'].unique(),fpl['player_name'].unique())
    fpl_matched = fpl[fpl['player_name'].isin(matching_names)]
    understat_matched = understat[understat['player_name'].isin(matching_names)]
    names_merged_on_date = pd.merge(fpl_matched, understat_matched, left_on=['player_name', 'kickoff_time'], right_on=['player_name', 'date']) 
    return names_merged_on_date, fpl_matched, understat_matched


def remove_matched_names(fpl, understat, names_merged_on_date):
    """[This function checks which names were matched previously, removes them and returns the unmatched data]

    Args:
        fpl ([type]): [description]
        understat ([type]): [description]
        names_merged_on_date ([type]): [description]

    Returns:
        [type]: [description]
    """    
    fpl_not_matched = fpl[~fpl['player_name'].isin(names_merged_on_date['player_name'].unique())] 
    understat_not_matched = understat[~understat['player_name'].isin(names_merged_on_date['player_name'].unique())] 
    return fpl_not_matched, understat_not_matched
    
    
season = '2019-20' 
# season = '2020-21'
fpl_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//gws//' 
understat_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//understat//'
data_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//'
understat, fpl_to_understat, fpl, teams, fixtures = read_data(fpl_path, understat_path)
fpl = merge_fpl_data(fpl, fixtures, teams) # No understat statistics included in this dataframe
fpl = rename_fpl_teams(fpl) # Keep team naming convention standard across understat and fpl
names_merged_on_date, fpl_matched, understat_matched = exact_matches(understat, fpl) # Data merged on player name and match date
fpl_not_matched, understat_not_matched = remove_matched_names(fpl, understat, names_merged_on_date) # Those names that did not match previously




# %%
understat_not_matched['player_name'].unique().__len__()



# %%
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Now find the entries whos names and dates DID NOT match exactly and use a similarity finder to
# manually look for potential spelling errors
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


similarity_matched_df = get_matching_names(understat_not_matched['player_name'].unique(),
                                           fpl_not_matched['player_name'].unique()) # Returns a dataframe of matching names with a degree of similarity

# %%

def create_renaming_dict(similarity_matched_df, understat_not_matched, fpl_not_matched, season):
    """[This function uses the result from get_matching_names to construct a renaming dictionary, as well as subset the data]

    Args:
        similarity_matched_df ([type]): [description]
        understat_not_matched ([type]): [description]
        fpl_not_matched ([type]): [description]
    """    
    if season == '2020-21':
        wrongly_matched_names = ['Alisson', 'Allan', 'André Gomes', 'Bernard', 'Bernardo', 'Bernardo Silva', 'David Luiz', 'Ederson', 'Emerson', 
                            'Fabinho', 'Felipe Anderson', 'Fred',  'Hélder Costa', 'Joelinton', 'Jonny', 'Jorginho', 'Kepa', 'Lucas Moura', 'Raphinha', 
                            'Ricardo Pereira', 'Rodri', 'Rúben Dias','Rúben Vinagre', 'Semi Ajayi', 'Trézéguet', 'Vitinha', 'Willian'] 
    if season == '2019-20':
        wrongly_matched_names = ['Adrián','Alisson','André Gomes','Angelino', 'Bernard', 'Bernardo', 'Bernardo Silva','Borja Bastón', 
                                 'Chicharito','David Luiz','Ederson', 'Emerson', 'Fabinho', 'Felipe Anderson', 'Fred','Joelinton', 'Jonny',
                                 'Jorginho','Jota', 'Kepa','Kiko Femenía','Pedro', 'Ricardo Pereira', 'Rodri','Rúben Vinagre','Trézéguet','Wesley','Willian']
    
    similar_rename = similarity_matched_df[~similarity_matched_df['understat'].isin(wrongly_matched_names)] # Subset similarly matching names
    understat_no_similar = understat_not_matched[understat_not_matched['player_name'].isin(wrongly_matched_names)] # Subset Understat: No similar match
    understat_similar = understat_not_matched[~understat_not_matched['player_name'].isin(wrongly_matched_names)] # Subset Understat: Similar match

    fpl_similar = fpl_not_matched[fpl_not_matched['player_name'].isin(similar_rename['fpl'].unique())] # Subset FPL: Similar match
    fpl_no_similar = fpl_not_matched[~fpl_not_matched['player_name'].isin(similar_rename['fpl'].unique())] # Subset FPL: No similar match
    
    name_mapper = dict(zip(similar_rename['understat'], similar_rename['fpl'])) # Create dictionary to rename the similarly matched names # Creates missing values
    understat_similar['player_name'] = understat_similar['player_name'].map(name_mapper)  # Rename, note all the wrongly matched ones are removed from the dataset
    
    return understat_no_similar, understat_similar, fpl_similar, fpl_no_similar


understat_no_similar, understat_similar, fpl_similar, fpl_no_similar = create_renaming_dict(similarity_matched_df, understat_not_matched, fpl_not_matched, season)
# %%
understat_similar.isnull().sum()
# %%
names_merged_on_similarity = pd.merge(fpl_similar, understat_similar, left_on=['player_name', 'kickoff_time'], right_on=['player_name', 'date']) # Merge using player name and date of game













