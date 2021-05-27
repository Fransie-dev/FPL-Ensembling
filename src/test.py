
import asyncio
import json
import aiohttp
import understat
import pandas as pd
import nest_asyncio
from datetime import datetime
from time import mktime
import time
nest_asyncio.apply()
async def main():
    async with aiohttp.ClientSession() as session:
        a = understat.Understat(session)
        player = await a.get_league_players(
            "epl", 2019#,
            #player_name="Paul Pogba",
            #team_title="Manchester United"
        )
        print(json.dumps(player))
        return player
    
if '__name__' == '__main__':
    main()
    