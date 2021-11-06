# %%
import difflib
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def merge_fixtures(fpl_path, understat_path, data_path):
    """[Merges team and fixtures onto fpl. Slightly processes data.]

    Args:
        fpl_path ([type]): [description]
        understat_path ([type]): [description]
        data_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    understat = pd.read_csv(understat_path + 'all_understat_players.csv')
    understat['date']=pd.to_datetime(understat['date']).dt.date
           
    fixtures = pd.read_csv(data_path + 'fixtures.csv')
    fixtures = fixtures[['id', 'team_a', 'team_a_difficulty', 'team_h', 'team_h_difficulty']]
    fixtures.rename(columns={'id':'fixture'}, inplace=True)
    
    teams = pd.read_csv(data_path + 'teams.csv')
    teams = teams[['id', 'name', 'short_name', 'strength', 'strength_attack_away', 'strength_attack_home', 'strength_defence_away', 'strength_defence_home', 'strength_overall_away', 'strength_overall_home']]
  

    fpl = pd.read_csv(fpl_path + 'merged_gw.csv') 
    fpl.rename(columns = {'name':'player_name'},inplace = True) 
    fpl['kickoff_time'] = pd.to_datetime(fpl['kickoff_time']).dt.date
    fpl = pd.merge(fpl, fixtures, on='fixture') 
    fpl = pd.merge(fpl, teams, left_on='team', right_on='name')
    fpl = rename_teams(teams, fpl)
    
    return understat, fpl

def rename_teams(teams, fpl, cols = ['team', 'opponent_team', 'team_a', 'team_h']):
    """[This function replaces the integer value used to uniquely identify teams 
    with the FPL's associated string; 1 -> Manchester United, 2 -> Manchester City]

    Args:
        teams ([type]): [description]
        fpl ([type]): [description]
        cols (list, optional): [description]. Defaults to ['team','opponent_team', 'team_a', 'team_h'].

    Returns:
        [type]: [A converted dataframe]
    """
    fpl = fpl.copy()
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
    # print(len(list(set(a) & set(b))), 'unique and matching names between FPL and Understat')
    return list(set(a) & set(b))
    
def rename_fpl_teams(fpl, features = ['team', 'team_a', 'team_h']):
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
    understat_names, fpl_names = understat_names['player_name'].unique(), fpl_names['player_name'].unique()
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
    similarity_matched_df = pd.DataFrame({'understat':understat_similar, 'fpl':fpl_similar, 'similarity': ratio}).copy()
    similarity_matched_df_final = similarity_matched_df.loc[similarity_matched_df.groupby('understat')['similarity'].idxmax()].copy()
    # print(similarity_matched_df_final.sort_values('similarity',ascending=False).to_latex())
    return similarity_matched_df_final

def exact_matches(understat, fpl, join = 'inner'):
    """[This function performs the first initial match, that is the entries whos names and dates match exactly between the
    understat and fpl datasets]

    Args:
        understat ([type]): [description]
        fpl ([type]): [description]

    Returns:
        [type]: [The merged data, and the sets of data used to construct it]
    """    
    matching_names = intersect(understat['player_name'].unique(), fpl['player_name'].unique())
    fpl_matched = fpl[fpl['player_name'].isin(matching_names)]
    understat_matched = understat[understat['player_name'].isin(matching_names)]
    exact_merge = pd.merge(fpl_matched, understat_matched, left_on=['player_name', 'kickoff_time'], right_on=['player_name', 'date'], how= join) 
    return exact_merge

def remove_matched_names(fpl, understat, exact_merge):
    """[This function checks which names were matched in the first name/date match performed, removes them and returns the 
    entries who were not matched]

    Args:
        fpl ([type]): [description]
        understat ([type]): [description]
        exact_merge ([type]): [description]

    Returns:
        [type]: [description]
    """    
    fpl_not_matched = fpl[~fpl['player_name'].isin(exact_merge['player_name'].unique())] 
    understat_not_matched = understat[~understat['player_name'].isin(exact_merge['player_name'].unique())] 
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
    if season == '2020-21' or '2021-22':
        wrongly_matched_names = ['Adrián', 'Alisson', 'Allan', 'André Gomes', 'Bernard', 'Bernardo', 'Bernardo Silva', 'David Luiz', 'Ederson', 'Emerson', 
                            'Fabinho', 'Felipe Anderson', 'Fred',  'Hélder Costa', 'Joelinton', 'Jonny', 'Jorginho', 'Kepa', 'Lucas Moura', 'Raphinha', 
                            'Ricardo Pereira', 'Rodri', 'Rúben Dias','Rúben Vinagre', 'Semi Ajayi', 'Trézéguet', 'Wesley', 'Willian'] 
    if season == '2019-20':
        wrongly_matched_names = ['Adrián','Alisson','André Gomes','Angelino', 'Bernard', 'Bernardo', 'Bernardo Silva','Borja Bastón', 
                                 'Chicharito','David Luiz','Ederson', 'Emerson', 'Fabinho', 'Felipe Anderson', 'Fred','Joelinton', 'Jonny',
                                 'Jorginho','Jota', 'Kepa','Kiko Femenía','Pedro', 'Ricardo Pereira', 'Rodri','Rúben Vinagre','Trézéguet','Wesley','Willian']

        
    
    similar_rename = similarity_matched_df[~similarity_matched_df['understat'].isin(wrongly_matched_names)] # Subset Similar: Similar match
    # no_similar_rename = similarity_matched_df[similarity_matched_df['understat'].isin(wrongly_matched_names)] # Subset Similar: Similar match
    # print(similar_rename.to_latex())
    # print(no_similar_rename.to_latex())
    understat_no_similar = understat_not_matched[understat_not_matched['player_name'].isin(wrongly_matched_names)] # Subset Understat: No similar match
    understat_similar = understat_not_matched[~understat_not_matched['player_name'].isin(wrongly_matched_names)] # Subset Understat: Similar match

    fpl_similar = fpl_not_matched[fpl_not_matched['player_name'].isin(similar_rename['fpl'].unique())] # Subset FPL: Similar match
    fpl_no_similar = fpl_not_matched[~fpl_not_matched['player_name'].isin(similar_rename['fpl'].unique())] # Subset FPL: No similar match
    
    name_mapper = dict(zip(similar_rename['understat'], similar_rename['fpl'])) 
    understat_similar['player_name'] = understat_similar['player_name'].map(name_mapper)  # Renames similarly matched names
    return understat_no_similar, understat_similar, fpl_similar, fpl_no_similar


def final_rename(understat_no_similar, fpl_no_similar, join = 'inner'):
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
    manual_merge = pd.merge(fpl_no_similar, understat_no_similar, left_on=['player_name', 'kickoff_time'],
                                     right_on=['player_name', 'date'], how=join) # Merge using player name and date of game
    return manual_merge

def final_merge_understat(exact_merge, similar_merge, manual_merge, understat):
    """[This function merges the three matches performed]

    Args:
        exact_merge ([type]): [description]
        similar_merge ([type]): [description]
        manual_merge ([type]): [description]

    Returns:
        [type]: [description]
    """    
    understat_final = pd.concat([exact_merge, similar_merge, manual_merge])
    # print('Exact: ', exact_merge.shape[0], ' instances merged corresponding to: ', exact_merge['player_name'].unique().__len__(), ' unique players')
    # print('Similar: ', similar_merge.shape[0], ' instances merged corresponding to: ', similar_merge['player_name'].unique().__len__(), ' unique players')
    # print('Manually: ', manual_merge.shape[0], ' instances merged corresponding to: ', manual_merge['player_name'].unique().__len__(), ' unique players')
    # print('Lost: ', understat.shape[0] -  understat_final.shape[0], ' instances corresponding to: ', understat['player_name'].unique().__len__() 
    #       -  understat_final['player_name'].unique().__len__(), ' unique players', end='\n\n')
    # print(list(set(understat_final['player_name'].unique()) - set(understat['player_name'].unique())))
    # print(understat_final['player_name'].unique().isin(understat['player_name'].unique()))
    return understat_final

def join_data(fpl, understat, season):
    
    exact_merge = exact_matches(understat, fpl) # Data merged on player name and match date
    fpl_not_matched, understat_not_matched = remove_matched_names(fpl, understat, exact_merge) # Those names that did not match previously
    
    similarity_matched_df = get_matching_names(understat_not_matched, fpl_not_matched) 
    understat_no_similar, understat_similar, fpl_similar, fpl_no_similar = map_similar_names(similarity_matched_df, understat_not_matched, fpl_not_matched, season)  # Note: Manual investigation
    # print(understat_no_similar, understat_similar)
    similar_merge = pd.merge(fpl_similar, understat_similar, left_on=['player_name', 'kickoff_time'], right_on=['player_name', 'date']) 
    
    no_similar_matches_df = get_matching_names(understat_no_similar, fpl_no_similar) # Note: Manual investigation
    manual_merge = final_rename(understat_no_similar, fpl_no_similar)
    understat_final = final_merge_understat(exact_merge, similar_merge, manual_merge, understat)
    
    return fpl, understat_final


def main(season):
    fpl_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//gws//' 
    understat_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//understat//'
    data_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//'
    training_path = f'C://Users//jd-vz//Desktop//Code//data//{season}//training//'
    understat, fpl = merge_fixtures(fpl_path, understat_path, data_path)
    fpl, understat = join_data(fpl, understat, season)
    fpl.to_csv(training_path + 'fpl.csv', index = False)
    understat.to_csv(training_path + 'understat_merged.csv', index = False)


if __name__ == "__main__":
    main(season='2021-22') # Successful execution
    # main(season='2020-21') # Successful execution
    # main(season='2019-20') # Successful execution
    print('Success!')
# %%
