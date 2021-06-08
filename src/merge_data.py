import difflib
import numpy as np
import pandas as pd
from collect_data import dlt_create_dir


def read_data(fpl_path, understat_path, data_path):
    """[This function reads in the data from collect_data.py and returns the unmerged understat and fpl data,
    including FPL team and fixture data]

    Args:
        fpl_path ([type]): [description]
        understat_path ([type]): [description]

    Returns:
        [understat]: ['goals','shots','xG','time','position','h_team','a_team','h_goals','a_goals','date','id','season','roster_id','xA','assists','key_passes','npg','npxG' 'xGChain' 'xGBuildup', 'player_name']
        [fpl]: [FPL api data]

    """
    understat = pd.read_csv(understat_path + 'all_understat_players.csv')
    understat['date']=pd.to_datetime(understat['date']).dt.date
           
    fpl = pd.read_csv(fpl_path + 'merged_gw.csv') 
    fpl.rename(columns = {'element':'FPL_ID', 'name':'player_name'},inplace = True) 
    fpl['kickoff_time'] = pd.to_datetime(fpl['kickoff_time']).dt.date
    
    fixtures = pd.read_csv(data_path + 'fixtures.csv')
    fixtures = fixtures[['id', 'team_a', 'team_a_difficulty', 'team_h', 'team_h_difficulty']]
    fixtures.rename(columns={'id':'fixture'}, inplace=True)

    
    teams = pd.read_csv(data_path + 'teams.csv')
    teams = teams[['id', 'name', 'short_name', 'strength', 'strength_attack_away', 'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 'strength_overall_home']]

    
    return understat, fpl, teams, fixtures

def rename_teams(teams,fpl, cols = ['team','opponent_team', 'team_a', 'team_h']):
    """[This function replaces the integer value used to uniquely identify teams 
    with the FPL's associated string; 1 -> Manchester United, 2 -> Manchester City]

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
    """[This function finds the intersection between two player name columns]

    Args:
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [The intersection]
    """    
    print(len(list(set(a) & set(b))), 'unique and matching names between FPL and Understat')
    return list(set(a) & set(b))

def union(a, b):
    """[This function finds the union between two player name columns]

    Args:
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [The union]
    """    
    print(len(list(set(a) | set(b))), 'unique, not necessarily matching entries within the two columns')
    return list(set(a) | set(b))
    

def merge_fpl_data(fpl, fixtures, teams):
    """[This function merges the player, team and fixture data scraped from the FPL API]

    Args:
        fpl ([type]): [description]
        fixtures ([type]): [description]
        teams ([type]): [description]

    Returns:
        [type]: [A merged dataframe]
    """    
    fpl = pd.merge(fpl, fixtures, on='fixture') 
    fpl = pd.merge(fpl, teams, left_on='team', right_on='name')
    fpl = rename_teams(teams, fpl)
    return fpl
    
def rename_fpl_teams(fpl, features = ['team', 'team_a', 'team_h', 'opponent_team', 'name']):
    """[This function replaces the acronyms used to indicate teams by the FPL API with the teams full names, as seen in the understat data]

    Args:
        fpl ([type]): [description]

    Returns:
        [type]: [A renamed dataframe]
    NOTE:
        New teams from different seasons need to be added here
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
    for feature in features:
        fpl[feature] = fpl[feature].map(team_reps)

    return fpl


def get_matching_names(understat_names, fpl_names):
    """[This function checks for similarity between understat and fpl names, and returns a dataframe with all the unique understat names with the most similarly 
    matching FPL name.]

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
    """[This function performs the first initial match, that is the entries whos names and dates match exactly between the
    understat and fpl datasets]

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
    """[This function checks which names were matched in the first name/date match performed, removes them and returns the 
    entries who were not matched]

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


def map_similar_names(similarity_matched_df, understat_not_matched, fpl_not_matched, season):
    """[This function performs the second match, by usin the similarity dataframe returned from get_matching_names. It is manually encoded to identify wrongly mapped names
    from each season which are removed from the dataset. The remaining understat names are used to rename the understat datasets player names
    and acronyms to the most logical and similar matching names]

    Args:
        similarity_matched_df ([type]): [description]
        understat_not_matched ([type]): [description]
        fpl_not_matched ([type]): [description]
    NOTE:
        New names from different seasons need to be added here
    """    
    if season == '2020-21':
        wrongly_matched_names = ['Adrián', 'Alisson', 'Allan', 'André Gomes', 'Bernard', 'Bernardo', 'Bernardo Silva', 'David Luiz', 'Ederson', 'Emerson', 
                            'Fabinho', 'Felipe Anderson', 'Fred',  'Hélder Costa', 'Joelinton', 'Jonny', 'Jorginho', 'Kepa', 'Lucas Moura', 'Raphinha', 
                            'Ricardo Pereira', 'Rodri', 'Rúben Dias','Rúben Vinagre', 'Semi Ajayi', 'Trézéguet', 'Wesley', 'Willian'] 
    if season == '2019-20':
        wrongly_matched_names = ['Adrián','Alisson','André Gomes','Angelino', 'Bernard', 'Bernardo', 'Bernardo Silva','Borja Bastón', 
                                 'Chicharito','David Luiz','Ederson', 'Emerson', 'Fabinho', 'Felipe Anderson', 'Fred','Joelinton', 'Jonny',
                                 'Jorginho','Jota', 'Kepa','Kiko Femenía','Pedro', 'Ricardo Pereira', 'Rodri','Rúben Vinagre','Trézéguet','Wesley','Willian']
    
    similar_rename = similarity_matched_df[~similarity_matched_df['understat'].isin(wrongly_matched_names)] # Subset Similar: Similar match
    
    understat_no_similar = understat_not_matched[understat_not_matched['player_name'].isin(wrongly_matched_names)] # Subset Understat: No similar match
    understat_similar = understat_not_matched[~understat_not_matched['player_name'].isin(wrongly_matched_names)] # Subset Understat: Similar match

    fpl_similar = fpl_not_matched[fpl_not_matched['player_name'].isin(similar_rename['fpl'].unique())] # Subset FPL: Similar match
    fpl_no_similar = fpl_not_matched[~fpl_not_matched['player_name'].isin(similar_rename['fpl'].unique())] # Subset FPL: No similar match
    
    name_mapper = dict(zip(similar_rename['understat'], similar_rename['fpl'])) 
    understat_similar['player_name'] = understat_similar['player_name'].map(name_mapper)  # Renames similarly matched names
    
    print(understat_similar['player_name'].unique().__len__(), 'unique similarly matched names\n') 
    print(understat_no_similar['player_name'].unique().__len__(), 'unique unmatched names\n')
    print(fpl_no_similar['player_name'].unique().__len__(), 'unique unmatched names\n')
    
    return understat_no_similar, understat_similar, fpl_similar, fpl_no_similar


def final_rename(understat_no_similar, fpl_no_similar):
    """[This function performs the third and final manual matching. It manually investigates those names that had no similar name, searches for the player name
    or nickname in the understat data, checks the team, Googles the player's true name, and finds the corresponding FPL name. The function then renames all those 
    understat entries to the associated FPL name]

    Args:
        understat_no_similar ([type]): [description]
        fpl_no_similar ([type]): [description]

    Returns:
        [type]: [description]
        
    NOTE:
        New names from different seasons need to be added here
    """    
    name_mapper = {'Adrián':'Adrián Bernabé', # Contains both seasons corrections
                   'Alisson':'Alisson Ramses Becker',
                   'Allan':'Allan Marques Loureiro',
                   'André Gomes':'André Filipe Tavares Gomes',
                   'Angelino':'José Ángel Esmorís Tasende',
                   'Bernard':'Bernard Anício Caldeira Duarte', # Everton
                   'Bernardo Silva':'Bernardo Mota Veiga de Carvalho e Silva', # Manchester City
                   'Bernardo':'Bernardo Fernandes da Silva Junior', # 
                   'Borja Bastón':'Borja González Tomás',
                   'Chicharito':'Javier Hernández Balcázar',
                   'David Luiz':'David Luiz Moreira Marinho', 
                   'Ederson':'Ederson Santana de Moraes',
                   'Emerson':'Emerson Palmieri dos Santos',
                   'Fabinho':'Fabio Henrique Tavares',
                   'Felipe Anderson':'Felipe Anderson Pereira Gomes',
                   'Fred':'Frederico Rodrigues de Paula Santos', # Manchester United
                   'Hélder Costa': 'Hélder Wander Sousa de Azevedo e Costa', # Leeds
                   'Joelinton':'Joelinton Cássio Apolinário de Lira', # Chelsea
                   'Jonny':'Jonathan Castro Otto', # Wolves
                   'Jorginho':'Jorge Luiz Frello Filho', # Chelsea
                   'Jota':'José Ignacio Peleteiro Romallo',
                   'Kepa':'Kepa Arrizabalaga',
                   'Kiko Femenía':'Francisco Femenía Far',
                   'Lucas Moura':'Lucas Rodrigues Moura da Silva',
                   'Pedro': 'Pedro Rodríguez Ledesma', # Chelsea
                   'Raphinha':'Raphael Dias Belloli',
                   'Ricardo Pereira':'Ricardo Domingos Barbosa Pereira',
                   'Rodri':'Rodrigo Hernandez',
                   'Rúben Dias':'Rúben Santos Gato Alves Dias',
                   'Rúben Vinagre':'Rúben Gonçalo Silva Nascimento Vinagre',
                   'Semi Ajayi':'Oluwasemilogo Adesewo Ibidapo Ajayi',
                   'Trézéguet':'Mahmoud Ahmed Ibrahim Hassan', # Aston Villa
                   'Wesley':'Wesley Moraes',
                   'Willian':'Willian Borges Da Silva',
                   }
    understat_no_similar['player_name'] = understat_no_similar['player_name'].map(name_mapper)
    names_merged_manually = pd.merge(fpl_no_similar, understat_no_similar, left_on=['player_name', 'kickoff_time'], right_on=['player_name', 'date']) # Merge using player name and date of game
    return names_merged_manually

def final_merge_understat(names_merged_on_date, names_merged_on_similarity, names_merged_manually):
    """[This function merges the three matches performed]

    Args:
        names_merged_on_date ([type]): [description]
        names_merged_on_similarity ([type]): [description]
        names_merged_manually ([type]): [description]

    Returns:
        [type]: [description]
    """    
    understat_final = pd.concat([names_merged_on_date, names_merged_on_similarity, names_merged_manually])
    return understat_final

def compare_datasets(understat_final, understat, fpl, data_path, season):
    """[This function compares the original and merged data, and saves the FPL and merged FPL/Understat data to CSV's.]

    Args:
        understat_final ([type]): [description]
        understat ([type]): [description]
        fpl ([type]): [description]
        data_path ([type]): [description]
    """    
    print('\nThe final merged understat data contains', understat_final.isnull().sum().sum(), 'missing values')
    print('The final merged understat data set contains', understat_final.__len__(), 'rows')
    print('The final merged understat data set contains', understat_final['player_name'].unique().__len__(), 'unique players')
    print('The original understat data set contains', understat_final.__len__(), 'rows')
    print('The original understat data set contains',understat['player_name'].unique().__len__(), 'unique players')
    print('The original FPL data set contains', fpl.__len__(), 'rows')
    print('The original FPL data set contains',fpl['player_name'].unique().__len__(), 'unique players\n')
    path = data_path + 'training//'
    dlt_create_dir(path)
    fpl.to_csv(path + 'fpl.csv', index = False)
    understat_final.to_csv(path + 'understat_merged.csv', index = False)
    
    


def main(season):
    fpl_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//gws//' 
    understat_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//understat//'
    data_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//'
    understat, fpl, teams, fixtures = read_data(fpl_path, understat_path, data_path)
    fpl = merge_fpl_data(fpl, fixtures, teams) # No understat statistics included in this dataframe
    fpl = rename_fpl_teams(fpl) # Keep team naming convention standard across understat and fpl
    names_merged_on_date, fpl_matched, understat_matched = exact_matches(understat, fpl) # Data merged on player name and match date
    fpl_not_matched, understat_not_matched = remove_matched_names(fpl, understat, names_merged_on_date) # Those names that did not match previously
    similarity_matched_df = get_matching_names(understat_not_matched['player_name'].unique(), fpl_not_matched['player_name'].unique()) # Returns a dataframe of matching names with a degree of similarity
    understat_no_similar, understat_similar, fpl_similar, fpl_no_similar = map_similar_names(similarity_matched_df, understat_not_matched, fpl_not_matched, season) # Maps similar names manually
    names_merged_on_similarity = pd.merge(fpl_similar, understat_similar, left_on=['player_name', 'kickoff_time'], right_on=['player_name', 'date']) # Merge using player name and date of game
    no_similar_matches_df = get_matching_names(understat_no_similar['player_name'].unique(), fpl_no_similar['player_name'].unique()) # Repeat process and manually investigate non-matching names
    names_merged_manually = final_rename(understat_no_similar, fpl_no_similar)
    understat_final = final_merge_understat(names_merged_on_date, names_merged_on_similarity, names_merged_manually)
    compare_datasets(understat_final, understat, fpl, data_path, season)


if __name__ == "__main__":
    main(season='2020-21') # Successful execution
    main(season='2019-20') # Successful execution
    print('Success!')


# TODO: Impute values in the merged understat data, such that more gameweeks are available.
