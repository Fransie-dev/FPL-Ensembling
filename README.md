# Fantasy Premier League Decision Support


## Project Description

This repository contains the source code for my mini-dissertation project, which is completed in partial fulfilment for the degree of Master of Industrial Engineering under the supervision of Dr. Thorsten Schmidt-Dumont. The aim of this mini-dissertation is to investigate the applicability of machine learning and optimisation techniques to suggest player transfers for a manager in the Fantasy Premier League. The final results obtained demonstrate that the implemented decision support system performs favorably, where the best set of suggested player transfers would have placed in the top _5.98%_ of eight million real-world managers' in the 2020/21 season. A substantial increase in performance was observed when removing all fixtures that occured before the pandemic, and testing the retrained models on the first seven gameweeks of the 2021/22 season.

## Project Goals

* Implement models from five distinctly functioning machine learning families and evaluate their performance.
* Implement a stacked ensemble of said models in an attempt to improve the performance.
* Implement a stacked ensemble of multiple models in an attempt to improve the performance.
* Implement model-averaging of the base- and meta-learners in an attempt to improve the performance.
* Formulate the team formulation problem as two linear sub-problems subject to team, position, transfer and cost constraints.
* Investigate how well the models would have performed in the 2020/21 Fantasy Premier League season
* Investigate the effect of using chips on the team performance.


## Credits
* Historical season data: https://github.com/vaastav/Fantasy-Premier-League
* Fantasy Premier League API: https://fantasy.premierleague.com/
* Understat API: https://understat.com/
