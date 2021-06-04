


""" 
matching_teams = intersect(fpl_20['team'], fpl_21['team'])
fpl_20_not_matched = fpl_20['team'][~fpl_20['team'].isin(matching_teams)].unique()
fpl_20_not_matched
# Do not match: array(['Watford', 'Bournemouth', 'Norwich'], dtype=object)
 """
# fpl_20['team'].unique()
# array(['West Ham', 'Man City', 'Burnley', 'Southampton', 'Brighton',
#        'Watford', 'Bournemouth', 'Sheffield Utd', 'Man Utd', 'Chelsea',
#        'Wolves', 'Leicester', 'Liverpool', 'Norwich', 'Everton',
#        'Crystal Palace', 'Aston Villa', 'Spurs', 'Arsenal', 'Newcastle'],
#       dtype=object)

# %%
""" fpl_21['team'].unique() 
fpl_21_not_matched = fpl_21['team'][~fpl_21['team'].isin(matching_teams)].unique()
fpl_21_not_matched
# Do not match: array(['Fulham', 'Leeds', 'West Brom'], dtype=object) """

# fpl_21['team'].unique()
# array(['Brighton', 'Chelsea', 'West Ham', 'Newcastle', 'Sheffield Utd',
#        'Wolves', 'Everton', 'Spurs', 'Fulham', 'Arsenal', 'Leeds',
#        'Liverpool', 'Leicester', 'West Brom', 'Southampton',
#        'Crystal Palace', 'Aston Villa', 'Man Utd', 'Man City', 'Burnley'],
#       dtype=object)

understat['h_team'].unique() # season = '2019-20' 
# %%


'West Ham', 'Man City', 'Sheffield Utd', 'Man Utd', 'Wolves', 'Spurs' # FPL 2019-20
'West Ham', 'Manchester City', 'Manchester United', 'Sheffield United', 'Wolverhampton Wanderers', 'Tottenham'

# Understat 2019

# %%
understat['h_team'].unique()
# %%
'West Ham', 'Man City', 'Sheffield Utd', 'Man Utd', 'West Brom', 'Spurs' # 2020-21
'West Ham', 'Manchester City', 'Manchester United', 'Sheffield United', 'West Bromwich Albion', 'Tottenham'


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
'Burnley':'Burnley'
# %%

len(team_reps)


# %%
intersect(fpl_20['team'], fpl_21['team'])
# %%
