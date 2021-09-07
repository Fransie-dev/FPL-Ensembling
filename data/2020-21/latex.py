# %%
import pandas as pd
elements = pd.read_csv('./players_raw.csv')
print(elements[['first_name', 'second_name', 'transfers_in', 'red_cards']].head(5).to_latex(index=False))
# %%
a
# %%
