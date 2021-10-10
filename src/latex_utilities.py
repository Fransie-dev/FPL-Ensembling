def check_shift():
    training_path = f'C://Users//jd-vz//Desktop//Code//data//2019-20//training//'
    df = pd.read_csv(training_path + 'cleaned_fpl.csv')
    df_shift = pd.read_csv(training_path + 'shifted_fpl.csv')
    a, b = df[df.player_name == 'Aaron Cresswell'][['GW', 'total_points', 'bps']], df_shift[df_shift.player_name == 'Aaron Cresswell'][['GW', 'total_points_shift', 'bps_shift']]
    print(pd.merge(a, b, on='GW'))


def check_dups():
    training_path = f'C://Users//jd-vz//Desktop//Code//data//2019-20//training//'
    df = pd.read_csv(training_path + 'cleaned_fpl.csv')
    df_shift = pd.read_csv(training_path + 'shifted_fpl.csv')
    print(df[df.duplicated(['player_name', 'GW'])])

# %%
import pandas as pd
training_path = 'C://Users//jd-vz//Desktop//Code//data//2020-21//'
df = pd.read_csv(training_path + 'players_raw.csv')
print(df[['first_name', 'second_name', 'id','team', 'element_type', 'goals_scored', 'total_points']].iloc[:,].to_latex())
# %%
training_path = 'C://Users//jd-vz//Desktop//Code//data//2020-21//players//Aaron_Connolly_78//'
df = pd.read_csv(training_path + 'gw.csv')
# %%
print(df[['element', 'fixture', 'kickoff_time', 'value', 'selected', 'total_points']].head().to_latex())

# %%
# %%
import pandas as pd
training_path = 'C://Users//jd-vz//Desktop//Code//data//'

def count_fixtures():
    df1 = pd.read_csv(training_path + 'collected_us.csv')
    import numpy as np
    df1['cnt'] = np.where(df1['was_home'], df1['kickoff_time'] + df1['team'] + df1['opponent_team'],  df1['kickoff_time'] + df1['opponent_team']  + df1['team'] )
    print(df1['cnt'].nunique())
    
count_fixtures()
    
# %%
import pandas as pd
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//collected_us.csv')
df['position_stat'].value_counts()
# %%
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//understat//all_understat_players.csv')

# %%
print(df.loc[[31,485, 419],['player_name', 'date', 'goals', 'xG', 'xG', 'npg', 'npxG' ]].to_latex())
# %%
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

# %%
new_name = [elem for elem in name_mapper.values()]
old_name = [elem for elem in name_mapper.keys()]
df = pd.DataFrame()
df['Understat'] = old_name
df['FPL'] = new_name
# %%
print(df.to_latex())
# %%

# %%
print(pd.DataFrame(name_mapper).to_latex())
# %%
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//understat//all_understat_players.csv')
print(df[['player_name', 'date', 'xG', 'npg', 'npxG' ]].to_latex())
# %%
list = ['Tottenham', 'Aston Villa', 'Wolverhampton Wanderers',
       'Sheffield United', 'Manchester United', 'Crystal Palace',
       'Burnley', 'West Bromwich Albion', 'Liverpool', 'Fulham', 'Leeds',
       'Leicester', 'Arsenal', 'Manchester City', 'Brighton', 'West Ham',
       'Newcastle United', 'Everton', 'Southampton', 'Chelsea', 'Reims',
       'Rennes', 'Monaco', 'Montpellier', 'Lorient', 'Bordeaux',
       'Saint-Etienne', 'Brest', 'Nimes', 'Dijon', 'Nice', 'Lyon',
       'Nantes', 'Real Sociedad', 'Osasuna', 'Atletico Madrid',
       'Villarreal', 'Granada', 'SD Huesca', 'Valencia', 'Celta Vigo',
       'Alaves', 'Real Madrid', 'Eintracht Frankfurt', 'Mainz 05',
       'Werder Bremen', 'FC Cologne', 'Hoffenheim', 'Schalke 04',
       'Borussia M.Gladbach', 'Bayer Leverkusen', 'VfB Stuttgart',
       'Angers', 'Hertha Berlin', 'Augsburg', 'Borussia Dortmund',
       'RasenBallsport Leipzig', 'Freiburg', 'Wolfsburg', 'AC Milan',
       'Cagliari', 'Benevento', 'Sassuolo', 'Genoa', 'Sampdoria',
       'Napoli', 'Crotone', 'Union Berlin', 'Arminia Bielefeld',
       'Paris Saint Germain', 'Lens', 'Lille', 'Strasbourg', 'Metz',
       'Marseille', 'Verona', 'Fiorentina', 'Udinese', 'Torino', 'Eibar',
       'Cadiz', 'Bayern Munich']

# %%
tms = ['Paris Saint Germain', 'Lens', 'Lille', 'Strasbourg', 'Metz', 'Marseille', 'Verona', 'Fiorentina']
print(df.loc[df['h_team'].isin(tms)][['player_name', 'date', 'xG', 'h_team', 'a_team']].to_latex())
# %%


df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2020-21//players_raw.csv')
df.head(10)





# %%

# sns.kdeplot(x = 'minutes', data = df)
# %%
sns.histplot(x = 'total_points', data = df, fill = True, hue = 'Season', palette = 'pastel', alpha = 0.9, multiple = 'stack', bins = 20)
