# %%
import pulp 
import pandas as pd
import numpy as np
import pulp 
import pandas as pd
import numpy as np

GWValues = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//stacked_predictions.csv')
all_teams = GWValues['team'].unique()
GWValues = GWValues[GWValues['GW'] == 1]
GWValues.shape # 274, 8
GWValues['value'] = GWValues['value']/10
# First of three models - page 35 to 36
# Selecting an initial squad of 15 players
model_1 = pulp.LpProblem("Model1",pulp.LpMaximize)
G = (GWValues['position']=='GK').astype(int).values # Binary values indicating indices of player positions
D = (GWValues['position']=='DEF').astype(int).values
M = (GWValues['position']=='MID').astype(int).values
F = (GWValues['position']=='FWD').astype(int).values    
# Define the parameter values for the team constraint
# If player i from team j then t_i_j = 1
length_opt = GWValues.shape[0]
# teams = GWValues['team'].unique() # TODO: Updt me.

t_i_j = [[0 for y in range(20)] for x in range(length_opt)]
for i in range(length_opt):
    for j in range(0,20):
        # if GWValues['team'].iloc[i] == teams[j]:#j + 1:
        if GWValues['team'].iloc[i] == all_teams[j]:#j + 1:
            t_i_j[i][j] = 1
t_i_j = np.array(t_i_j)
# Decision variables - eqn (4.3)
x_vars_model_1  = {i:pulp.LpVariable(cat=pulp.LpBinary, name=("x_{}".format(i))) for i in range(length_opt)}
# Dummy variable for money left over after initial squad selected
money_left_1 = pulp.LpVariable(cat=pulp.LpContinuous,lowBound=0, name = 'money_left_1')
# Objective Function - eqn (4.1) - Points here is a vector containing the points total for player i from the previous season
model_1 += pulp.lpSum(GWValues.iloc[i]['predicted_points']*x_vars_model_1[i] for i in range(length_opt)) , 'Objective Function'
# Constraints
model_1 += pulp.lpSum(GWValues.iloc[i]['value']*x_vars_model_1[i] for i in range(length_opt)) + money_left_1 == 100, 'Cost constraint' # eqn (4.4)
model_1 += pulp.lpSum(G[i]*x_vars_model_1[i] for i in range(length_opt)) == 2, 'Goalkeepers constraint' # eqn (4.5)
model_1 += pulp.lpSum(D[i]*x_vars_model_1[i] for i in range(length_opt)) == 5, 'Defenders constraint' # eqn (4.6)
model_1 += pulp.lpSum(M[i]*x_vars_model_1[i] for i in range(length_opt)) == 5, 'Midfielders constraint' # eqn (4.7)
model_1 += pulp.lpSum(F[i]*x_vars_model_1[i] for i in range(length_opt)) == 3, 'Forwards constraint' # eqn (4.8)
for j in range(20):
    model_1 += pulp.lpSum(t_i_j[i,j]*x_vars_model_1[i] for i in range(length_opt)) <= 3 # eqn (4.9) - Team constraint
# Perform the optimisation
model_1.solve()
# Extract the players selected
xvars_res = []
for i in range(length_opt):
    q = int(pulp.value(x_vars_model_1[i]))
    xvars_res.append(q)
xvars_res = np.array(xvars_res) # Note: Does not consist of binary values
a = GWValues.iloc[xvars_res ==1] # Note: Incomplete subset
m_gm1 = pulp.value(money_left_1) # Note: 0 money left?
# print(a['team'].value_counts().max()) #
# %%
GWValues['team'].unique().__len__()

# %%





# %% Second of three models

#Other variables - defined as above
G = (GWValues['position']=='GK').astype(int).values # Binary values indicating indices of player positions
D = (GWValues['position']=='DEF').astype(int).values
M = (GWValues['position']=='MID').astype(int).values
F = (GWValues['position']=='FWD').astype(int).values       
teams = GWValues['team']

t_i_j = [[0 for y in range(20)] for x in range(length_opt)]
for i in range(length_opt):
    for j in range(1,20):
        if GWValues['team'].iloc[i] == j + 1:
            t_i_j[i][j] = 1
t_i_j = np.array(t_i_j)

# Start defining the optimisation model
model_2 = pulp.LpProblem('Model2',pulp.LpMaximize)
m = 56 # An integer dummy variable used during the optimisationa - defined as M in (4.17) and (4.18)
optimization_predictions = GWValues['predicted_points'] # Get the points prediction values
allowed_aditional = 1 # The number of additional free transfers allowed


m_gm1 = float(np.array(report.iloc[(GW-2):(GW-1), 2:3])) # TODO: Indice is GW. Counter of money left.
xvars_res = x_final_chosen # TODO: Indices of final players chosen

# Decision variables
x_g = {i:pulp.LpVariable(cat=pulp.LpBinary, name=("x_g_{}".format(i))) for i in range(length_opt)} # x_i,g

money_left_2 = pulp.LpVariable(cat = pulp.LpContinuous, lowBound = 0, name = 'money left model 2') # The amount of money left after the week

# Additional decision variables
t_g = pulp.LpVariable(cat = pulp.LpInteger, name = 't_g', lowBound = 0) #The number of transfers performed
f_g = pulp.LpVariable(cat = pulp.LpInteger, name = 'f_g', lowBound = 0) #The number of free transfers available
y_model_2 = pulp.LpVariable(cat = pulp.LpBinary, name = 'y_model_2') # Dummy variable for the if-else constraints (4.17) and (4.18)
q_g = pulp.LpVariable(cat=pulp.LpContinuous, lowBound = 0, name = 'q_g') # Variable for the points penalty incurred due to more transfers than the number of free transfers being performed

# Constraints
model_2 += m_gm1 + pulp.lpSum(costs[i]*xvars_res[i] for i in range(length_opt)) == money_left_2 + pulp.lpSum(costs[i]*x_g[i] for i in range(length_opt)), 'Bookkeeping constraint' # Ensure budget is not exceeded - eqn (4.14)
model_2 += pulp.lpSum(G[i]*x_g[i] for i in range(length_opt)) == 2, 'Goalkeepers constraint'
model_2 += pulp.lpSum(D[i]*x_g[i] for i in range(length_opt)) == 5, 'Defenders constraint'
model_2 += pulp.lpSum(M[i]*x_g[i] for i in range(length_opt)) == 5, 'Midfielders constraint'
model_2 += pulp.lpSum(F[i]*x_g[i] for i in range(length_opt)) == 3, 'Forwards constraint'    
model_2 +=  f_g <= 1, 'Free trades constraint' 
model_2 += -q_g +4*(t_g-f_g) <= m*y_model_2, 'if' # eqn (4.17)
model_2 += t_g - f_g <= m*(1-y_model_2), 'else'  # eqn (4.18)
model_2 += 15-pulp.lpSum(xvars_res[i]*x_g[i] for i in range(length_opt)) == t_g, 'Exchanges' # eqn (4.19)
model_2 += q_g <= 4*allowed_aditional 

for j in range(20):
    model_2 += pulp.lpSum(t_i_j[i,j]*x_g[i] for i in range(length_opt)) <= 3 # Team constraint 

#Objective function
model_2 += pulp.lpSum(optimization_predictions[i]*x_g[i] for i in range(length_opt)) - q_g # eqn (4.16)

model_2.solve()

trades = pulp.value(t_g)
        
m_gm2 = pulp.value(money_left_2)

money_prev = m_gm2

penalty = pulp.value(q_g)

# %% Final model
# Model for selecting a starting 11
# Let's determine the starting eleven
model_3 = pulp.LpProblem('Model3', pulp.LpMaximize)
optimization_predictions = GWValues['predicted_points'] # et the points prediction values

# Decision variables
s_model_3  = {i:pulp.LpVariable(cat=pulp.LpBinary, name=("s_{}".format(i))) for i in range(length_opt)} # eqn (4.25)

# Constraints
model_3 += pulp.lpSum(s_model_3[i] for i in range(length_opt)) == 11, '11 constraint' # eqn (4.26)
model_3 += pulp.lpSum(G[i]*s_model_3[i] for i in range(length_opt)) == 1, 'GK constraint' # eqn (4.27)
model_3 += pulp.lpSum(D[i]*s_model_3[i] for i in range(length_opt)) >= 3, 'Defenders constraint' # eqn (4.28)
model_3 += pulp.lpSum(F[i]*s_model_3[i] for i in range(length_opt)) >= 1, 'Forwards constraint' # eqn (4.29)

# Objective function
model_3 += pulp.lpSum(optimization_predictions[i]*x_final_chosen[i]*s_model_3[i] for i in range(length_opt)), 'Objective function' # eqn (4.24)
# TODO: Indices Out of 15, choose 11
# Solve the model
model_3.solve()
# %%
