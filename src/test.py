def create_strength(fpl_df):
    """[This function merges a team's fixture statistics based on the was_home feature]

    Args:
        fpl_df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    
    strength_attack = []
    strength_defence = []
    strength_overall = []
    for i in range(len(fpl_df)):
        if fpl_df['was_home'][i] == True:
            strength_attack.append(fpl_df['strength_attack_home'][i])
            strength_defence.append(fpl_df['strength_defence_home'][i])
            strength_overall.append(fpl_df['strength_attack_home'][i]/2 + fpl_df['strength_defence_home'][i]/2) 
        if fpl_df['was_home'][i] == False:
            strength_attack.append(fpl_df['strength_attack_away'][i])
            strength_defence.append(fpl_df['strength_defence_away'][i])
            strength_overall.append(fpl_df['strength_attack_away'][i]/2 + fpl_df['strength_defence_away'][i]/2) 
    fpl_df['team_strength_attack'] = strength_attack
    fpl_df['team_strength_defence'] = strength_defence
    fpl_df['team_strength_overall'] = strength_overall
    fpl_df.drop(['strength','strength_attack_away','strength_attack_home','strength_defence_away','strength_defence_home','strength_overall_away','strength_overall_home'], axis = 1, inplace = True)
    return fpl_df


fpl_df = create_strength(fpl_df)