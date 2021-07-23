# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import pandas as pd
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('C://Users//jd-vz//Desktop//Code//data//2019-20//training//cleaned_imp.csv', index_col=0)

# %%
for column in df.select_dtypes("number").columns:
    df.pivot(columns="position")[column].plot.hist(alpha=0.5)
    plt.title(column)
    plt.show()
    
# %%
# for column in df.select_dtypes("object").columns.drop("position"):
#     df.pivot(columns="position")[column].apply(pd.value_counts).plot.bar()
#     plt.title(column)
#     plt.show()  