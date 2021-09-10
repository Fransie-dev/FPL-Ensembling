import matplotlib.pyplot as plt
import seaborn as sns
gb_sales2=df_test.groupby(by="position")[["total_points_shift"]].sum()
res_2=gb_sales2.reset_index()
plt.pie(x="total_points_shift",labels="position", data=df_test[df_test["total_points_shift"]>0])
plt.xlabel("Sales 2016")
plt.show()