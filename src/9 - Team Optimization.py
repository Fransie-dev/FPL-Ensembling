# %%
import pandas as pd
import pulp
import numpy as np
import sklearn.metrics as metrics
import requests
import pandas as pd
import seaborn as sns


def Access_URL(url):
    r = requests.get(url) 
    json = r.json() 
    return json 

def plot_top_managers(team_ids = range(1, 1000, 1)):
    sns.set()
    overall_dataframe = pd.DataFrame()
    for id in team_ids:
        previous_seasons = Access_URL(f'https://fantasy.premierleague.com/api/entry/{id}/history/')['past']
        df_player = pd.DataFrame(previous_seasons)
        df_player['id'] = id
        overall_dataframe = pd.concat([overall_dataframe, df_player])
    last_season = overall_dataframe.loc[overall_dataframe['season_name'] == '2020/21']
    last_season.plot(kind='scatter', x='total_points', y='rank', figsize=(10, 10))
    mask = (last_season['total_points'] >= 2400) 
    last_season.loc[mask].plot(kind='scatter', x='total_points', y='rank', figsize=(10, 10))
    
def regression_results(y_true, y_pred):
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    
def get_repeat(use):
    df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//model_predictions.csv')
    names = pd.DataFrame({'player_name': df['player_name'].unique()})
    df_repeated = pd.concat([names]*38, ignore_index=True) 
    df_repeated['GW'] = np.repeat(range(1, 39), names.shape[0]) # Contains all names duplicated 38 times
    df_repeated = pd.merge(df_repeated, df, on = ['player_name', 'GW'], how = 'outer') # Get player stats
    for col in ['total_points', use]:
        df_repeated[col] = df_repeated.groupby(['player_name', 'GW'])[col].transform('sum') # Merge double weeks
        df_repeated.drop_duplicates(['player_name', 'GW', col], inplace = True)
        df_repeated[col].fillna(0, inplace = True) # Dont want these players
    f = lambda x: x.median() if np.issubdtype(x.dtype, np.number) else x.mode().iloc[0]
    df_repeated = df_repeated.fillna(df_repeated.groupby('player_name').transform(f)) 
    df_repeated.sort_values(['GW', 'player_name']).reset_index(drop = True, inplace = True)
    return df_repeated

def select_team(expected_scores, prices, positions, clubs, total_budget=100, sub_factor=0.2):
    """[Function to pick a starting XV]

    Args:
        expected_scores ([type]): [description]
        prices ([type]): [description]
        positions ([type]): [description]
        clubs ([type]): [description]
        total_budget (int, optional): [description]. Defaults to 100.
        sub_factor (float, optional): [description]. Defaults to 0.2.

    Returns:
        [type]: [description]
    """    
    num_players = len(expected_scores)
    model = pulp.LpProblem("Constrained_Value_Maximisation", pulp.LpMaximize)
    decisions = [pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer') for i in range(num_players)]
    captain_decisions = [pulp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer') for i in range(num_players)]
    sub_decisions = [pulp.LpVariable("z{}".format(i), lowBound=0, upBound=1, cat='Integer') for i in range(num_players)]

    # objective function:
    model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * expected_scores[i] for i in range(num_players)), "Objective"

    # cost constraint
    model += sum((decisions[i] + sub_decisions[i]) * prices[i] for i in range(num_players)) <= total_budget  # total cost

    # position constraints
    # 1 starting goalkeeper
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'GK') == 1
    # 2 total goalkeepers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 'GK') == 2

    # 3-5 starting defenders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'DEF') >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'DEF') <= 5
    
    # 5 total defenders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 'DEF') == 5

    # 3-5 starting midfielders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'MID') >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'MID') <= 5
    
    # 5 total midfielders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 'MID') == 5

    # 1-3 starting attackers
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'FWD') >= 1
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'FWD') <= 3
    # 3 total attackers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 'FWD') == 3

    # club constraint
    for club_id in np.unique(clubs):
        model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players

    model += sum(decisions) == 11  # total team size
    model += sum(captain_decisions) == 1  # 1 captain
    
    for i in range(num_players):  
        model += (decisions[i] - captain_decisions[i]) >= 0  # captain must also be on team
        model += (decisions[i] + sub_decisions[i]) <= 1  # subs must not be on team

    model.solve()
    # print("Total expected score = {}".format(model.objective.value()))
    return decisions, captain_decisions, sub_decisions


def get_starting_XV(df_repeated, GW = 1, max_budget = 100):
    df =  df_repeated[df_repeated['GW'] == GW].reset_index(drop = True)
    expected_scores = df[use] 
    actual_scores = df['total_points']
    prices = df["value"] /10
    positions = df["position"]
    clubs = df["team"]
    decisions, captain_decisions, sub_decisions = select_team(expected_scores.values, prices.values, positions.values, clubs.values, total_budget=max_budget)
    squad_indices, starting_indices, sub_indices, captain_indice = [],[],[],[]
    print(f'Gameweek: {GW}')
    for i in range(len(decisions)):
        if decisions[i].value() == 1:
            if captain_decisions[i].value() == 1:
                captain_indice.append(i) 
            squad_indices.append(i)
            starting_indices.append(i)
    for i in range(len(sub_decisions)):
        if sub_decisions[i].value() == 1:
            squad_indices.append(i)
            sub_indices.append(i)
    actual_max_budget = 100
    budget = round(actual_max_budget - prices.iloc[squad_indices].sum(), 2)
    exp_captain_point = expected_scores.iloc[captain_indice].sum()*2 if GW in TRIPLE_CAPTAIN else expected_scores.iloc[captain_indice].sum()
    act_captain_point = actual_scores.iloc[captain_indice].sum()*2 if GW in TRIPLE_CAPTAIN else actual_scores.iloc[captain_indice].sum()
    actual_points = round(actual_scores.iloc[squad_indices].sum() + act_captain_point, 2) if GW in BENCH_BOOST else round(actual_scores.iloc[starting_indices].sum() +  act_captain_point, 2)
    ex_pts = round(expected_scores.iloc[squad_indices].sum() + exp_captain_point, 2) if GW in BENCH_BOOST else round(expected_scores.iloc[starting_indices].sum()  + exp_captain_point, 2)
    # print(f'Starting budget: {max_budget}')
    # print('Budget left:', budget, end = '\n')
    # print(f'Expected scored = {expected_scores.iloc[starting_indices].sum()} + {expected_scores.iloc[captain_indice].sum()} = {ex_pts}') 
    # print(f'Points scored = {actual_scores.iloc[starting_indices].sum()}  + {actual_scores.iloc[captain_indice].sum()} = {actual_points}') 
    return squad_indices, starting_indices, sub_indices, captain_indice, budget, actual_points

def get_decision_array(name, length):
    return np.array([
        pulp.LpVariable("{}_{}".format(name, i), lowBound=0, upBound=1, cat='Integer')
        for i in range(length)
    ])

class TransferOptimiser:
    def __init__(self, expected_scores, buy_prices, sell_prices, positions, clubs):
        self.expected_scores = expected_scores
        self.buy_prices = buy_prices
        self.sell_prices = sell_prices
        self.positions = positions
        self.clubs = clubs
        self.num_players = len(expected_scores)

    def instantiate_decision_arrays(self):
        # we will make transfers in and out of the squad, and then pick subs and captains from that squad
        transfer_in_decisions_free = get_decision_array("transfer_in_free", self.num_players)
        transfer_in_decisions_paid = get_decision_array("transfer_in_paid", self.num_players)
        transfer_out_decisions = get_decision_array("transfer_out_paid", self.num_players)
        # total transfers in will be useful later
        transfer_in_decisions = transfer_in_decisions_free + transfer_in_decisions_paid
        sub_decisions = get_decision_array("subs", self.num_players)
        captain_decisions = get_decision_array("captain", self.num_players)
        return transfer_in_decisions_free, transfer_in_decisions_paid, transfer_out_decisions, transfer_in_decisions, sub_decisions, captain_decisions

    def encode_player_indices(self, indices):
        decisions = np.zeros(self.num_players)
        decisions[indices] = 1
        return decisions

    def apply_transfer_constraints(self, model, transfer_in_decisions_free, transfer_in_decisions, transfer_out_decisions, budget_now):
        # only 1 free transfer
        model += sum(transfer_in_decisions_free) <= 1
        # budget constraint
        transfer_in_cost = sum(transfer_in_decisions * self.buy_prices)
        transfer_out_cost = sum(transfer_out_decisions * self.sell_prices)
        budget_next_week = budget_now + transfer_out_cost - transfer_in_cost
        model += budget_next_week >= 0

    def solve(self, current_squad_indices, budget_now, sub_factor, GW):
        current_squad_decisions = self.encode_player_indices(current_squad_indices)

        model = pulp.LpProblem("Transfer_Optimisation", pulp.LpMaximize)
        transfer_in_decisions_free, transfer_in_decisions_paid, transfer_out_decisions, transfer_in_decisions, sub_decisions, captain_decisions = self.instantiate_decision_arrays()

        # calculate new team from current team + transfers
        next_week_squad = current_squad_decisions + transfer_in_decisions - transfer_out_decisions
        starters = next_week_squad - sub_decisions

        # points penalty for additional transfers
        transfer_penalty = sum(transfer_in_decisions_paid) if GW in WILDCARD or FREE_HIT else sum(transfer_in_decisions_paid) * 4 # 4-pt incremental hit

        self.apply_transfer_constraints(model, transfer_in_decisions_free, transfer_in_decisions, transfer_out_decisions, budget_now)
        self.apply_formation_constraints(model, squad=next_week_squad, starters=starters, subs=sub_decisions, captains=captain_decisions)

        # objective function:
        model += self.get_objective(starters, sub_decisions, captain_decisions, sub_factor, transfer_penalty, self.expected_scores), "Objective"
        status = model.solve()

        return transfer_in_decisions, transfer_out_decisions, starters, sub_decisions, captain_decisions

    def get_objective(self, starters, subs, captains, sub_factor, transfer_penalty, scores):
        """[Maximize points scored by starters, substitutions, captains subtracted by the transfer penalty]

        Args:
            starters ([type]): [description]
            subs ([type]): [description]
            captains ([type]): [description]
            sub_factor ([type]): [description]
            transfer_penalty ([type]): [description]
            scores ([type]): [description]

        Returns:
            [type]: [description]
        """        
        starter_points = sum(starters * scores)
        sub_points = sum(subs * scores) * sub_factor
        captain_points = sum(captains * scores) # Does not need to account for triple captain
        return starter_points + sub_points + captain_points - transfer_penalty # Accounts for doubles

    def apply_formation_constraints(self, model, squad, starters, subs, captains):
        position_data = {
                        "gk": {"position_id": 'GK', "min_starters": 1, "max_starters": 1, "num_total": 2},
                        "df": {"position_id": 'DEF', "min_starters": 3, "max_starters": 5, "num_total": 5},
                        "mf": {"position_id": 'MID', "min_starters": 3, "max_starters": 5, "num_total": 5},
                        "fw": {"position_id": 'FWD', "min_starters": 1, "max_starters": 3, "num_total": 3}
                        }
        for position, data in position_data.items():
            # formation constraints
            model += sum(starter for starter, position in zip(starters, self.positions) if position == data["position_id"]) >= data["min_starters"]
            model += sum(starter for starter, position in zip(starters, self.positions) if position == data["position_id"]) <= data["max_starters"]
            model += sum(selected for selected, position in zip(squad, self.positions) if position == data["position_id"]) == data["num_total"]

        # club constraint
        for club_id in np.unique(self.clubs):
            model += sum(selected for selected, club in zip(squad, self.clubs) if club == club_id) <= 3  # max 3 players

        # total team size
        model += sum(starters) == 11
        model += sum(squad) == 15
        model += sum(captains) == 1

        for i in range(self.num_players):
            model += (starters[i] - captains[i]) >= 0  # captain must also be on team
            model += (starters[i] + subs[i]) <= 1  # subs must not be on team

def manage_squad(df_repeated, GW, squad_indices, budget = 0):
    df =  df_repeated[df_repeated['GW'] == GW].reset_index(drop = True)
    actual_scores = df['total_points']
    expected_scores = df[use]
    prices = df["value"] /10
    old_price = round(prices.iloc[squad_indices].sum(), 2)
    positions = df["position"]
    clubs = df["team"]
    opt = TransferOptimiser(expected_scores.values, prices.values, prices.values, positions.values, clubs.values)
    transfer_in_decisions, transfer_out_decisions, decisions, sub_decisions, captain_decisions = opt.solve(squad_indices, budget_now=budget, sub_factor=0.2, GW = GW)
    squad_indices, starting_indices, sub_indices, captain_indice, p_out, p_in = [], [], [], [], [], []
    print(f'Gameweek: {GW}', end = '\n')
    for i in range(len(decisions)):
        if decisions[i].value() == 1:
            squad_indices.append(i)
            starting_indices.append(i)
            if captain_decisions[i].value() == 1:
                captain_indice.append(i)
    for i in range(len(sub_decisions)):
        if sub_decisions[i].value() == 1:
            squad_indices.append(i)
            sub_indices.append(i)
    for i in range(len(transfer_out_decisions)):
        if transfer_out_decisions[i].value() == 1:
            p_out.append(prices.iloc[i])
    for i in range(len(transfer_in_decisions)):
        if transfer_in_decisions[i].value() == 1:
            p_in.append(prices.iloc[i])
    penalty = 0 if GW in WILDCARD or FREE_HIT else ((len(p_in) - 1) * 4)
    exp_captain_point = expected_scores.iloc[captain_indice].sum()*2 if GW in TRIPLE_CAPTAIN else expected_scores.iloc[captain_indice].sum()
    act_captain_point = actual_scores.iloc[captain_indice].sum()*2 if GW in TRIPLE_CAPTAIN else actual_scores.iloc[captain_indice].sum()
    actual_points = round(actual_scores.iloc[squad_indices].sum() - penalty + act_captain_point, 2) if GW in BENCH_BOOST else round(actual_scores.iloc[starting_indices].sum() - penalty + act_captain_point, 2)
    ex_pts = round(expected_scores.iloc[squad_indices].sum() - penalty + exp_captain_point, 2) if GW in BENCH_BOOST else round(expected_scores.iloc[starting_indices].sum() - penalty + exp_captain_point, 2)
    budget_new = round(old_price - prices.iloc[squad_indices].sum() + budget, 2)
    # print('Starting budget', budget)
    # print('Transferred', len(p_out), 'players out for ', '+', round(sum(p_out), 2))
    # print('Transferred', len(p_in), 'players in at ', '-', round(sum(p_in), 2))
    # print(f'Expected scored = {expected_scores.iloc[starting_indices].sum()} - {penalty} + {exp_captain_point} = {ex_pts}') 
    # print(f'Points scored = {actual_scores.iloc[starting_indices].sum()} - {penalty} + {act_captain_point} = {actual_points}') 
    # print(f'Budget left = {old_price} - {round(prices.iloc[squad_indices].sum(), 2)} + {budget} = {budget_new}', end = '\n')
    return squad_indices, starting_indices, sub_indices, captain_indice, budget_new, actual_points

def format_squad(df_repeated, starting_indices, captain_indice, sub_indices, actual_points):
    starters = df_repeated[df_repeated['GW'] == GW].reset_index(drop = True).iloc[starting_indices] 
    captain = df_repeated[df_repeated['GW'] == GW].reset_index(drop = True).iloc[captain_indice] 
    subs = df_repeated[df_repeated['GW'] == GW].reset_index(drop = True).iloc[sub_indices] 
    subs['sub'], starters['starter'], captain['captain']  = 1, 1, 1
    starters = pd.merge(starters, captain[['player_name', 'captain']], on = 'player_name', how = 'outer')
    squad = pd.concat([starters, subs])
    squad = squad.fillna(0).sort_values(['captain', 'starter', 'position', 'total_points'], ascending=False).reset_index(drop = True)
    squad['base_points'] = np.where(squad['starter'] == 1, squad.groupby('GW')['total_points'].transform('sum'), 0) 
    squad['wildcard'] = np.where(squad['GW'].isin(WILDCARD), 1, 0)
    squad['triple_captain'] = np.where(squad['GW'].isin(TRIPLE_CAPTAIN), 1, 0)
    squad['bench_boost'] = np.where(squad['GW'].isin(BENCH_BOOST), 1, 0)
    squad['actual_points'] = actual_points
    return squad

use = 'total_points' 
total_points, squads = [], []
df_repeated = get_repeat(use) # Replicate all players`
WILDCARD = [18, 36] # Cannot use in GW 1
TRIPLE_CAPTAIN = [1]
BENCH_BOOST = [1]
FREE_HIT = [29] # Cannot use in GW 1
BUDGET = 100
for GW in range(1,39):
    if GW in FREE_HIT:
        reset_squad_indices, reset_starting_indices, reset_sub_indices, reset_captain_indice = squad_indices, starting_indices, sub_indices, captain_indice 
    # Main Loop
    if GW == 1:
        squad_indices, starting_indices, sub_indices, captain_indice, budget, actual_points =  get_starting_XV(df_repeated, max_budget = BUDGET, GW = GW) # For the first gameweek
    else:
        squad_indices, starting_indices, sub_indices, captain_indice, budget, actual_points = manage_squad(df_repeated, GW, squad_indices, budget)
    squad = format_squad(df_repeated, starting_indices, captain_indice, sub_indices, actual_points)
    squads.append(squad)
    total_points.append(actual_points)
    if GW in FREE_HIT:
        squad_indices, starting_indices, sub_indices, captain_indice = reset_squad_indices, reset_starting_indices, reset_sub_indices, reset_captain_indice 
print(f'Total season points (including penalty, double captain, and chips): {sum(total_points)}')
df = pd.concat(squads) # 2245 sonder wildcards
# %%# %%
# Chip strategy
# df[['GW', 'actual_points']].drop_duplicates().sort_values('actual_points') 
# Iter 1: Gameweek 18 gets worst score -- Use WILDCARD 
# Iter 2: Gameweek 36 does the worst -- Use WILDCARD
# Iter 3: Gameweek 1 Captain does the best -- Use TRIPLE_CAPTAIN
# Iter 4: Gameweek 1 Bench does the best -- Use BENCH_BOOST
# Iter 5: Gameweek 29 does the worst -- Use FREEHIT