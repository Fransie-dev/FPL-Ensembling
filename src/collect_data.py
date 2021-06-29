import os
import pandas as pd
import requests
from utilities import dlt_create_dir
import csv
import shutil
import asyncio
import aiohttp
from understat import Understat 
import pandas as pd
import nest_asyncio
from datetime import datetime
from time import mktime
import time
import pandas as pd
import progressbar as pb
nest_asyncio.apply()
pbar = pb.ProgressBar()

def Access_URL(url):
    """[
        This function simply sends a GET request to the URL]

    Args:
        url ([type]): [The URL to access]

    Returns:
        [JSON]: [Returns the json-encoded content of the response]
    """    
    r = requests.get(url) 
    json = r.json() 
    return json 

def Get_FPL_Data(path):
    """[
        This function collects the relevant current season's data from the Bootstrap API and writes them to CSV files
        players_raw.csv - Current Season Statistics per Player
        players_type.csv - Team Constraints and Conversion For Element_Types in Elements_DF
        teams.csv - Team codes and statistics]

    Args:
        path ([type]): [The location where to save the resulting CSV file]
    """    
    json = Access_URL(url = 'https://fantasy.premierleague.com/api/bootstrap-static/')
    elements_df = pd.DataFrame(json['elements'])  # Player statistics
    elements_types_df = pd.DataFrame(json['element_types']) # Rules/positions
    teams_df = pd.DataFrame(json['teams']) # Statistics/team
    elements_df.to_csv(path + 'players_raw.csv', index = False) 
    elements_types_df.to_csv(path + 'players_type.csv', index = False)
    teams_df.to_csv(path + 'teams.csv', index = False)
    json = Access_URL(url = 'https://fantasy.premierleague.com/api/fixtures/')
    fixtures_df = pd.DataFrame(json)  
    fixtures_df.to_csv(path + 'fixtures.csv', index = False)

def Get_Player_Historic_Data(data_path, player_history_path):
    """[
        This function writes a CSV file for each player which contains 
        some averaged statistics regarding the players past season histories [/history.csv] [2015/16 - 2019/20].
        The functions also writes a CSV file for the players' current season gameweek history [/gw.csv] [2020/2021]

    Args:
        player_history_path ([type]): [The directory of the data]
        players_data ([type]): [The elements_df/players_raw.csv from Get_FPL_Data]
    
    Inspiration for this approach:
        @github.com/ritviyer/
        

    """    
    players = os.listdir(player_history_path) # Lists All The Player Folders in the Dir
    players_data = pd.read_csv(data_path + 'players_raw.csv', index_col=0)
    for ind in pbar(players_data.index): # ind in [0:693:1]
    # Get the Seasonal History
        player_path = players_data['first_name'][ind] + '_' + players_data['second_name'][ind] + '_' + str(players_data['id'][ind]) # Create player_history_path
        if player_path not in players: # If the player (read from players_raw.csv) is not within the existing directory, continue: 
            json = Access_URL(url = "https://fantasy.premierleague.com/api/element-summary/{}/".format(str(players_data['id'][ind]))) # Feed in Player ID
            # print(json.keys())
            history_df = pd.DataFrame(json['history_past']) # Extract history
            if not history_df.empty: # If history returned
                os.makedirs(player_history_path + player_path, exist_ok = True) # Create a new path for the player  
                history_df.to_csv(player_history_path + player_path + '/history.csv', encoding='utf-8', index = False) # And write his syeasonal history
        else: # However, if the player is within the existing directory
            if not os.path.isfile(player_history_path + player_path + "/history.csv"): # And a history file does not exist
                json = Access_URL(url = "https://fantasy.premierleague.com/api/element-summary/{}/".format(str(players_data['id'][ind]))) # Feed in Player ID
                history_df = pd.DataFrame(json['history_past']) # Extract history
                if not history_df.empty: # If history returned
                    history_df.to_csv(player_history_path + player_path + '/history.csv', encoding='utf-8', index = False) # And write his seasonal history
    # Get the Gameweek History
        json = Access_URL(url = "https://fantasy.premierleague.com/api/element-summary/{}/".format(str(players_data['id'][ind]))) # Feed in Player ID 
        history_df_gw = pd.DataFrame(json['history']) # Extract Gameweek History
        if not history_df_gw.empty: # If history returned
            if player_path not in players: # If the player (read from players_raw.csv) is not within the existing directory, continue: 
                os.makedirs(player_history_path + player_path, exist_ok = True) # Create the directory, exit
            history_df_gw.to_csv(player_history_path + player_path + '/gw.csv', encoding='utf-8', index = False) # Write the CSV



def get_teams(directory):
    """[This function returns FPL team data]

    Args:
        directory ([type]): [Where the FPL CSV is written]
        
    Credit:
        https://github.com/vaastav/Fantasy-Premier-League/blob/master/collector.py
    """    
    teams = {}
    fin = open(directory + "/teams.csv", 'rU')
    reader = csv.DictReader(fin)
    for row in reader:
        teams[int(row['id'])] = row['name']
    return teams


def get_fixtures(directory):
    """[This function returns FPL fixture data]

    Args:
        directory ([type]): [Where the FPL CSV is written]
        
    Credit:
        https://github.com/vaastav/Fantasy-Premier-League/blob/master/collector.py
    """    
    fixtures_home = {}
    fixtures_away = {}
    fin = open(directory + "/fixtures.csv", 'rU')
    reader = csv.DictReader(fin)
    for row in reader:
        fixtures_home[int(row['id'])] = int(row['team_h'])
        fixtures_away[int(row['id'])] = int(row['team_a'])
    return fixtures_home, fixtures_away


def get_positions(directory):
    """[This function returns FPL position data]

    Args:
        directory ([type]): [Where the FPL CSV is written]
        
    Credit:
        https://github.com/vaastav/Fantasy-Premier-League/blob/master/collector.py
    """    
    positions = {}
    names = {}
    pos_dict = {'1': "GK", '2': "DEF", '3': "MID", '4': "FWD"}
    fin = open(directory + "/players_raw.csv", 'rU',encoding="utf-8")
    reader = csv.DictReader(fin)
    for row in reader:
        positions[int(row['id'])] = pos_dict[row['element_type']] 
        names[int(row['id'])] = row['first_name'] + ' ' + row['second_name']
    return names, positions

def collect_gw(gw, gameweek_path, data_path, player_path):
    """[This function scrapes the FPL data and collects data for a ]

    Args:
        directory ([type]): [Where the FPL CSV is written]
        
    Credit:
        https://github.com/vaastav/Fantasy-Premier-League/blob/master/collector.py
    """    
    rows = []
    fieldnames = []
    fixtures_home, fixtures_away = get_fixtures(data_path)
    teams = get_teams(data_path)
    names, positions = get_positions(data_path)
    for root, dirs, files in os.walk(player_path):
        for fname in files:
            if fname == 'gw.csv':
                fpath = os.path.join(root, fname)
                fin = open(fpath, 'rU')
                reader = csv.DictReader(fin)
                fieldnames = reader.fieldnames
                for row in reader:
                    if int(row['round']) == gw:
                        id = int(os.path.basename(root).split('_')[-1])
                        name = names[id]
                        position = positions[id]
                        fixture = int(row['fixture'])
                        if row['was_home'] == True or row['was_home'] == "True":
                            row['team'] = teams[fixtures_home[fixture]]
                        else:
                            row['team'] = teams[fixtures_away[fixture]]
                        row['name'] = name
                        row['position'] = position
                        rows += [row]

    fieldnames = ['name', 'position', 'team'] + fieldnames
    outf = open(os.path.join(gameweek_path, "gw" + str(gw) + ".csv"), 'w', encoding="utf-8")
    writer = csv.DictWriter(outf, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
        
def collect_all_gw(max_gw, gameweek_path, data_path, player_path):
    """[This function recursively calls collect all_gw]

    Args:
        max_gw ([type]): [description]
        gameweek_path ([type]): [description]
        data_path ([type]): [description]
        player_path ([type]): [description]
    """    
    for i in list(range(1, max_gw + 1)): # Check here
        collect_gw(i, gameweek_path=gameweek_path, data_path=data_path, player_path=player_path)
    merge_gw(type='FPL', gameweek_path=gameweek_path)
    


def count_directory(gws_folder):
    """[This function lists the number of gameweek files that can be merged within a directory]

    Args:
        path ([type]): [The folder where the gameweeks are ]

    Returns:
        [type]: [The number of gameweeek files to merge]
    """
    count_num_gw = 0
    for file in os.listdir(gws_folder):
        if file.startswith('gw'): # eg: '.txt'
            count_num_gw += 1
    print(f'There are {count_num_gw} gameweek files that can be merged')
    if count_num_gw == 47:
        count_num_gw = delete_directory(gws_folder)
        print(f'After deleting, there are now {count_num_gw} gameweek files that can be merged')
    else:
        print('No issues; continuing to merge gameweek files')
    return count_num_gw

def delete_directory(gws_folder):
    """[This function is a hard-coded solution to a read-in error within the 2019 data, where gaemweeks 30 - 38 contain no data, yet 39 - 47 do.]

    Args:
        gws_folder ([type]): [description]
    """    
    if os.path.exists(gws_folder + 'gw47.csv'):
        for gw in range(30, 39):
            os.remove(gws_folder + f'gw{gw}.csv')
            os.rename(src=gws_folder + f'gw{gw+9}.csv',
                    dst=gws_folder + f'gw{gw}.csv')
    count = count_directory(gws_folder)
    return count

def merge_gw(type, gameweek_path):  
    """[This function scans through the directory of different gameweek histories, and merges them 
    into one file]

    Args:
        gameweek_path ([type]): [Where the gameweek files are listed]
        type: 'FPL' or 'Understat'
    
    Inspiration:
        https://github.com/vaastav/Fantasy-Premier-League/blob/master/collector.py
    """ 
    count_directory(gameweek_path)
    if type == 'Understat':
        prefix = 'US_gw'
    if type == 'FPL':
        prefix = 'gw'
    num_gws = 38
    filepath = gameweek_path + f'merged_{prefix}.csv'
    if os.path.exists(filepath):
        os.remove(filepath)
    for gw in range(1, num_gws + 1): # + 1 because range is exclusive
        merged_gw_filename = f"merged_{prefix}.csv" # Output file
        gw_filename = prefix + str(gw) + ".csv" 
        gw_path = os.path.join(gameweek_path, gw_filename)
        fin = open(gw_path, encoding="utf-8")
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        fieldnames += ["GW"]
        rows = []
        for row in reader:
            row["GW"] = gw
            rows += [row]
        out_path = os.path.join(gameweek_path, merged_gw_filename)
        fout = open(file = out_path,
                    mode='a', 
                    encoding="utf-8")
        writer = csv.DictWriter(fout, fieldnames=fieldnames, lineterminator='\n')
        if gw == 1:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f'Succesfully wrote the {prefix} gameweek files from {gameweek_path} into a merged gameweek file to {out_path}')
    

    
async def get_league_players(season):
    """[This function is used to get a list of all the players]

    Returns:
        [type]: [description]
    """    
    if season == '2020-21':
        get_epl = 2020
    if season == '2019-20':
        get_epl = 2019
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        player = await understat.get_league_players("epl", get_epl)
        # print(json.dumps(player))
        return player
    
async def get_player_history(understat_id):
    """[This function is used to get a history of the player id returned from get_league_players]

    Args:
        understat_id ([type]): [description]

    Returns:
        [type]: [description]
    """    
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        player_matches = await understat.get_player_matches(understat_id)
        #print(json.dumps(player_matches))
        return player_matches
    
def write_league_players(understat_path, season):
    """[This function calls get_leauge_players and writes the resulting file to a csv]

    Args:
        understat_path ([type]): [description]

    Returns:
        [type]: [description]
    """    
    loop = asyncio.get_event_loop()
    players = loop.run_until_complete(get_league_players(season))
    player = pd.DataFrame.from_dict(players) # Equivalent of players_raw.csv
    player.to_csv(understat_path + 'understat_players.csv', index = False)
    return player

def set_season_time(season):
    if season == '2020-21':
        startdate = time.strptime('12-08-2020', '%d-%m-%Y')
        startdate = datetime.fromtimestamp(mktime(startdate))
        enddate = time.strptime('26-07-2021', '%d-%m-%Y')
        enddate = datetime.fromtimestamp(mktime(enddate))
    if season == '2019-20':
        startdate = time.strptime('09-08-2019', '%d-%m-%Y')
        startdate = datetime.fromtimestamp(mktime(startdate))
        enddate = time.strptime('26-07-2020', '%d-%m-%Y')
        enddate = datetime.fromtimestamp(mktime(enddate))
    return startdate, enddate
        

def get_all_player_history(understat_path, season):
    """[summary]

    Args:
        players ([type]): [The result of write_league_players]
    """    

    start_date, end_date = set_season_time(season)
    players = write_league_players(understat_path, season)
    # for i in pbar(list(range(len(players)))): # Throws out of range error in 2020-21 season
    for i in range(len(players)):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(get_player_history(int(players.loc[i][0])))
        name = players.loc[i][1]
        individuals = pd.DataFrame.from_dict(result)
        individuals['date'] = pd.to_datetime(individuals['date'])
        individuals = individuals[(individuals.date >= start_date)]
        individuals = individuals[(individuals.date <= end_date)]
        individuals['player_name'] = name
        individuals.to_csv(understat_path + "{}_data.csv".format(name), index = False) 
        if i == 0:
            all_players = individuals
        else:
            all_players = all_players.append(individuals)
    all_players.to_csv(understat_path + 'all_understat_players.csv', index = False) 

def main(season):
    if season == '2020-21':
        # General data
        data_path = 'C://Users//jd-vz//Desktop//Code//data//2020-21//'
        print(f'General data location: {data_path}', end = '\n')
        inp_general = input('Delete existing general data and download again? [y/n]\n')
        if inp_general == 'y':
            dlt_create_dir(data_path)
            Get_FPL_Data(path=data_path)
            print('Success', end = '\n')
        
        # Detailed player data   
        player_path = 'C://Users//jd-vz//Desktop//Code//data//2020-21//players//'
        print(f'Player data location: {player_path}', end = '\n')
        inp_detailed = input('Delete existing player data and download again? [y/n]\n')
        if inp_detailed == 'y':
            dlt_create_dir(player_path)
            Get_Player_Historic_Data(data_path,player_path)
            print('Success', end = '\n')
        
        # Scrape for and merge gamweek data
        gameweek_path = 'C://Users//jd-vz//Desktop//Code//data//2020-21//gws//'  
        print(f'Gameweek data location: {gameweek_path}', end = '\n')
        inp_gameweek = input('Delete existing gameweek data and scrape for it again? [y/n]\n') # Error
        if inp_gameweek ==  'y':
            dlt_create_dir(gameweek_path)
            collect_all_gw(max_gw=38, gameweek_path=gameweek_path, data_path=data_path, player_path=player_path)
            print('Success', end = '\n')
            
        # Understat data
        understat_path = 'C://Users//jd-vz//Desktop//Code//data//2020-21//understat//'
        print(f'Understat data location: {understat_path}', end = '\n')
        inp_gameweek = input('Delete existing understat data and scrape for it again? [y/n]\n')
        if inp_gameweek ==  'y':
            dlt_create_dir(understat_path)
            get_all_player_history(understat_path, season) # Written as understat_players.csv
            print('Success', end = '\n')
            
    if season == '2019-20':
        #! Copy data from vaastav before continuing
        # General data
        data_path = 'C://Users//jd-vz//Desktop//Code//data//2019-20//'
        print(f'General data location: {data_path}', end = '\n')
        
        # Detailed player data   
        player_path = 'C://Users//jd-vz//Desktop//Code//data//2019-20///players//'
        print(f'Player data location: {player_path}', end = '\n')
        
        # Scrape for and merge gamweek data
        gameweek_path = 'C://Users//jd-vz//Desktop//Code//data//2019-20//gws//'  
        print(f'Gameweek data location: {gameweek_path}', end = '\n')
        inp_gameweek = input('Delete existing gameweek data and scrape for it again? [y/n]\n')
        if inp_gameweek ==  'y':
            dlt_create_dir(gameweek_path)
            collect_all_gw(max_gw=47, gameweek_path=gameweek_path, data_path=data_path, player_path=player_path)
            print('Success', end = '\n')
            
        # Understat data
        understat_path = 'C://Users//jd-vz//Desktop//Code//data//2019-20//understat//'
        print(f'Understat data location: {understat_path}', end = '\n')
        inp_gameweek = input('Delete existing understat data and scrape for it again? [y/n]\n')
        if inp_gameweek ==  'y':
            dlt_create_dir(understat_path)
            get_all_player_history(understat_path, season) # Written as understat_players.csv
            print('Success', end = '\n')
        
    
        

        
                
    

if __name__ == "__main__":
    main(season='2020-21') # Successful execution
    main(season='2019-20') # Successful execution
    print('Success!')
