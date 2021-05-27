# %%
import asyncio
import aiohttp
from understat import Understat 
from collect_data import dlt_create_dir
import pandas as pd
import nest_asyncio
from datetime import datetime
from time import mktime
import time
import pandas as pd
import progressbar as pb
nest_asyncio.apply()
pbar = pb.ProgressBar()

# %%
async def get_league_players():
    """[This function is used to get a list of all the players]

    Returns:
        [type]: [description]
    """    
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        player = await understat.get_league_players("epl", 2020)
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
    
def write_league_players(understat_path):
    """[This function calls get_leauge_players and writes the resulting file to a csv]

    Args:
        understat_path ([type]): [description]

    Returns:
        [type]: [description]
    """    
    loop = asyncio.get_event_loop()
    players = loop.run_until_complete(get_league_players())
    player = pd.DataFrame.from_dict(players) # Equivalent of players_raw.csv
    player.to_csv(understat_path + 'understat_players.csv')
    return player

def get_all_player_history(players):
    """[summary]

    Args:
        players ([type]): [The result of get_league_players]
    """    
    startdate = time.strptime('12-08-2020', '%d-%m-%Y')
    startdate = datetime.fromtimestamp(mktime(startdate))
    enddate = time.strptime('26-07-2021', '%d-%m-%Y')
    enddate = datetime.fromtimestamp(mktime(enddate))
    for i in pbar(list(range(len(players)))):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(main(int(players.loc[i][0])))
        name = players.loc[i][1]
        individuals = pd.DataFrame.from_dict(result)
        individuals['date'] = pd.to_datetime(individuals['date'])
        individuals = individuals[(individuals.date >= startdate)]
        individuals = individuals[(individuals.date <= enddate)]
        individuals['player_name'] = name
        individuals.to_csv("./{}_data.csv".format(name))
        if i == 0:
            all_players = individuals
        else:
            all_players = all_players.append(individuals)

# %%
def main():
    understat_path = 'C://Users//jd-vz//Desktop//Code//data//understat//'
    dlt_create_dir(understat_path)
    players = write_league_players(understat_path) # Written as understat_players.csv

        
    
# %%
if __name__ == '__main__':
    main()