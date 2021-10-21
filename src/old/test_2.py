# thanks to https://github.com/joconnor-ml/forecasting-fantasy-football
import math
import re

# https://github.com/joconnor-ml/forecasting-fantasy-football
import pulp


"""
Returns a PuLP model containing the FPL objective function and starting xi,
position count, team count, captaincy count, and sub count constraints. Also
returns an unsolved starting xi decisions list, substitute 1 decisions list,
substitute 2 decisions list, substitute 3 decisions list, and captaincy
decisions list. Model does not include the cost constraint.
Takes in lists of players, their respective teams, their respective predicted
points, their respective positions, the number of
captains, and a tuple/list of size 3 in form [s1, s2, s3], where s1, s2, and
s3 are the probabilities that the first sub, second sub, and third sub will
play respectively.
"""


def base_lp_model(players, teams, points, positions, num_captains,
                  sub_factors):
    model = pulp.LpProblem("Constrained value maximisaiton", pulp.LpMaximize)
    num_players = len(players)
    starting_decisions = [
        pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    captain_decisions = [
        pulp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    sub_1_decision = [
        pulp.LpVariable("z{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    sub_2_decision = [
        pulp.LpVariable("2{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    sub_3_decision = [
        pulp.LpVariable("3{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    # objective function:
    model += sum((captain_decisions[i] * (1.0 / num_captains) +
                  starting_decisions[i] +
                  sub_1_decision[i] * sub_factors[0] +
                  sub_2_decision[i] * sub_factors[1] +
                  sub_3_decision[i] * sub_factors[2]) * points[i]
                 for i in range(num_players)), "Objective"
    # 1 starting gk
    model += sum(starting_decisions[i] for i in range(num_players)
                 if positions[i] == 1) == 1
    # No GK Subs
    model += sum(sub_1_decision[i] + sub_2_decision[i] + sub_3_decision[i]
                 for i in range(num_players) if positions[i] == 1) == 0
    # 3-5 starting defenders
    model += sum(starting_decisions[i] for i in range(num_players)
                 if positions[i] == 2) >= 3
    model += sum(starting_decisions[i] for i in range(num_players)
                 if positions[i] == 2) <= 5
    # 5 total defenders
    model += sum(starting_decisions[i] + sub_1_decision[i] +
                 sub_2_decision[i] + sub_3_decision[i]
                 for i in range(num_players) if positions[i] == 2) == 5
    # 3-5 starting midfielders
    model += sum(starting_decisions[i] for i in range(num_players)
                 if positions[i] == 3) >= 3
    model += sum(starting_decisions[i] for i in range(num_players)
                 if positions[i] == 3) <= 5
    # 5 total midfielders
    model += sum(starting_decisions[i] + sub_1_decision[i] +
                 sub_2_decision[i] + sub_3_decision[i]
                 for i in range(num_players) if positions[i] == 3) == 5
    # 1-3 starting strikers
    model += sum(starting_decisions[i] for i in range(num_players)
                 if positions[i] == 4) >= 1
    model += sum(starting_decisions[i] for i in range(num_players)
                 if positions[i] == 4) <= 3
    # 5 total strikers
    model += sum(starting_decisions[i] + sub_1_decision[i] +
                 sub_2_decision[i] + sub_3_decision[i]
                 for i in range(num_players) if positions[i] == 4) == 3
    # No more than 3 players from each club
    for team in teams:
        model += sum(starting_decisions[i] + sub_1_decision[i] +
                     sub_2_decision[i] + sub_3_decision[i]
                     for i in range(num_players) if teams[i] == team) <= 3
    for i in range(num_players):
        # captain must also be on team
        model += (starting_decisions[i] - captain_decisions[i]) >= 0
        # subs must not be on team
        model += (starting_decisions[i] + sub_1_decision[i]) <= 1
        model += (starting_decisions[i] + sub_2_decision[i]) <= 1
        model += (starting_decisions[i] + sub_3_decision[i]) <= 1
        # subs must be unique
        model += (sub_2_decision[i] + sub_1_decision[i]) <= 1
        model += (sub_3_decision[i] + sub_2_decision[i]) <= 1
        model += (sub_1_decision[i] + sub_3_decision[i]) <= 1
    model += sum(starting_decisions) == 11  # 11 starters
    model += sum(captain_decisions) == num_captains  # num_captains captains
    # 3 subs
    model += sum(sub_1_decision) == 1
    model += sum(sub_2_decision) == 1
    model += sum(sub_3_decision) == 1
    return model, starting_decisions, sub_1_decision, sub_2_decision, \
        sub_3_decision, captain_decisions


def lp_team_select(past_gameweeks, future_gameweeks, weights=None, budget=96,
                   min_mins=4*60, num_gws=4, refresh_data=False,
                   sub_factors=[0.3, 0.2, 0.1], num_captains=2, blacklisted=[]):
    strengths = team_strengths(past_gameweeks, weights, refresh_data=refresh_data)
    players, teams, points, positions, prices, next_week = get_data(past_gameweeks, future_gameweeks,
                                                                    strengths, min_mins,num_gws, refresh_data=refresh_data, blacklisted=blacklisted)
    num_players = len(players)
    model, starting_decisions, sub_1_decision, sub_2_decision, \
        sub_3_decision, captain_decisions = \
        base_lp_model(players, teams, points, positions, num_captains,
                      sub_factors)
    # max cost constraint
    model += sum((starting_decisions[i] + sub_1_decision[i] + sub_2_decision[i]
                  + sub_3_decision[i]) * prices[i]
                 for i in range(num_players)) <= budget  # total cost
    model.solve()
    return players, starting_decisions, sub_1_decision, sub_2_decision, \
        sub_3_decision, captain_decisions, next_week


"""
Given a current_team which is a list of 14 Player objects(exclude backup gk),
a list of their current selling costs, a maximum number of transfer_count, and
the amount of money in the bank(itb) suggests which players to remove.
Arguments are largely the same as lp_team_select except itb replaces budget.
"""


def lp_transfer(current_team, selling_prices, transfer_count, past_gameweeks,
                future_gameweeks, weights=None, itb=0, min_mins=4*60,
                num_gws=4, refresh_data=False, sub_factors=[0.3, 0.2, 0.1],
                num_captains=1, blacklisted=[]):
    budget = sum(selling_prices) + itb
    strengths = \
        team_strengths(past_gameweeks, weights, refresh_data=refresh_data)
    players, teams, points, positions, prices, next_week =  \
        get_data(past_gameweeks, future_gameweeks, strengths, min_mins,
                 num_gws, blacklisted=blacklisted)
    num_players = len(players)
    # Change prices for players already in current_team because of the way
    # FPL handles players' selling prices after price rises.
    for i in range(num_players):
        for j in range(len(current_team)):
            if players[i] == current_team[j]:
                prices[i] = selling_prices[j]
    model, starting_decisions, sub_1_decision, sub_2_decision, \
        sub_3_decision, captain_decisions = \
        base_lp_model(players, teams, points, positions, num_captains,
                      sub_factors)
    # max cost constraint
    model += sum((starting_decisions[i] + sub_1_decision[i] + sub_2_decision[i]
                  + sub_3_decision[i]) * prices[i]
                 for i in range(num_players)) <= budget  # total cost
    # max transfer_count constraints
    model += sum(starting_decisions[i] + sub_1_decision[i] +
                 sub_2_decision[i] + sub_3_decision[i]
                 for i in range(num_players) if players[i] in current_team) \
        >= len(current_team) - transfer_count
    model.solve()
    return players, starting_decisions, sub_1_decision, sub_2_decision, \
        sub_3_decision, captain_decisions, next_week


def find_weights(size, factor, step_size):
    return [factor ** (i // step_size)
            for i in range(size // step_size * step_size + step_size)][-size:]


"""
Pick team and print it nicely.
past_gws - number of past gameweeks to consider when judging player ability
future_gws - number of gameweeks to forecast for
min_mins and num_gws - player must have played min_mins over num_gws
refresh_data - whether or not to grab data from the fpl api
sub_factors - the probability each sub comes on
num_captains - number of estimated players to share the armband
"""


def pick_team(cur_gw, past_gws=10, future_gws=10, budget=96, min_mins=240,
              num_gws=4, refresh_data=False, sub_factors=[.2, .1, .05],
              num_captains=2, weights=None, blacklisted=[]):
    if cur_gw <= past_gws:
        past_gws = cur_gw - 1
    if weights is None:
        weights = find_weights(past_gws, 2, 5)
    players, starting, sub1, sub2, sub3, captain, next_week = \
        lp_team_select(range(cur_gw - past_gws, cur_gw),
                       range(cur_gw, cur_gw+future_gws),
                       weights=weights, budget=budget,
                       min_mins=min_mins, num_gws=num_gws,
                       refresh_data=refresh_data, sub_factors=sub_factors,
                       num_captains=num_captains, blacklisted=blacklisted)
    next_week = [next_week[i] for i in range(len(players)) if
                 starting[i].value() != 0 or sub1[i].value() != 0 or
                 sub2[i].value() != 0 or sub3[i].value() != 0]
    players = [players[i] for i in range(len(players)) if
               starting[i].value() != 0 or sub1[i].value() != 0 or
               sub2[i].value() != 0 or sub3[i].value() != 0]

    starting_xi, subs, captains = pick_xi(players, next_week)
    print(f'gw {cur_gw} starters:')
    print_xi(starting_xi)
    print('')
    print('Subs:')
    print(subs)
    print('')
    print('Captains:')
    print(captains)


"""
Suggests transfers and prints them nicely
Team is a list of players who are in the current team
prices is a list of team's respective prices
transfer_count is the number of transfers we are allowed to make
itb is the amount left in the bank
The rest of the params are the same as pick_team
"""


def transfer(cur_gw, team, prices, transfer_count, past_gws=9, future_gws=10,
             itb=0, min_mins=240, num_gws=4, refresh_data=False,
             sub_factors=[.2, .1, .05], num_captains=2, weights=None, blacklisted=[]):
    if weights is None:
        weights = find_weights(past_gws, 2, 5)
    players, starting, sub1, sub2, sub3, captain, next_week = \
        lp_transfer(team, prices, transfer_count, range(cur_gw - past_gws,
                                                        cur_gw),
                    range(cur_gw, cur_gw+future_gws),
                    weights=weights, itb=itb,
                    min_mins=min_mins, num_gws=num_gws,
                    refresh_data=refresh_data, sub_factors=sub_factors,
                    num_captains=num_captains, blacklisted=blacklisted)
    starting_xi = [players[i] for i in range(len(players)) if
                   starting[i].value() != 0]
    next_week_prev_team = [next_week[i] for i in range(len(players)) if
                           players[i] in team]
    next_week = [next_week[i] for i in range(len(players)) if
                 starting[i].value() != 0 or sub1[i].value() != 0 or
                 sub2[i].value() != 0 or sub3[i].value() != 0]
    n_week = [players[i] for i in range(len(players)) if
                 starting[i].value() != 0 or sub1[i].value() != 0 or
                 sub2[i].value() != 0 or sub3[i].value() != 0]
    players = [players[i] for i in range(len(players)) if
               starting[i].value() != 0 or sub1[i].value() != 0 or
               sub2[i].value() != 0 or sub3[i].value() != 0]
    starting_xi, subs, captains = pick_xi(players, next_week)
    print(f'gw {cur_gw} starters:')
    print_xi(starting_xi)

    print('')
    print('Subs:')
    print(subs)
    print('')
    print('Captains:')
    print(captains)
    new_team = list(starting_xi) + list(subs)
    print('\n Buy:')
    print([i for i in new_team if i not in team])
    print('\n Sell:')
    print([i for i in team if i not in new_team])


"""
Players - a list of player objects
next_week - a list of each player's respective points
Returns the optimal starting xi, the subs in order, and the captain pick
"""


def pick_xi(players, next_week):
    teams = [i.team for i in players]
    positions = [i.position for i in players]
    model, starting_decisions, sub_1_decision, sub_2_decision, \
        sub_3_decision, captain_decision = \
        base_lp_model(players, teams, next_week, positions, 1,
                      sub_factors=[0, 0, 0])
    model.solve()
    starters = [players[i] for i in range(len(players))
                if starting_decisions[i].value() != 0]
    starter_points = [next_week[i] for i in range(len(players))
                      if starting_decisions[i].value() != 0]
    # Order starters by points so we can pick captain
    starter_points, starters = zip(*sorted(zip(starter_points,
                                               starters), reverse=True))
    subs = list()
    sub_points = list()
    for sub in (sub_1_decision, sub_2_decision, sub_3_decision):
        subs += [players[i] for i in range(len(players))
                 if sub[i].value() != 0]
        sub_points += [next_week[i] for i in range(len(players))
                       if sub[i].value() != 0]
    # order the subs according to their points
    sub_points, subs = zip(*sorted(zip(sub_points, subs), reverse=True))
    captains = starters[:2]
    return starters, subs, captains


def print_xi(xi):
    team = {1: list(), 2: list(), 3: list(), 4: list()}
    for player in xi:
        team[player.position].append(player)
    for position in team:
        print(team[position])