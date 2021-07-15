library(ggplot2)
setwd('C://Users//jd-vz//Desktop//Code//misc')
df <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//data//2019-20//training//cleaned_fpl.csv')
fpl_feat <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//src//models//lin_reg//misc//selected_fpl_features.csv')
new_df <- subset (df, select = c(goals_scored,goals_conceded,assists,clean_sheets,penalties_saved,penalties_missed,saves,own_goals,red_cards,yellow_cards,minutes,value,bonus,bps,transfers_in,influence,creativity,threat,ict_index,position))


pdf("plots.pdf")
print(GGally::ggpairs(new_df, aes(colour = position, alpha = 0.6), title="Pairs plot for FPL dataset") + theme_grey(base_size = 10))
dev.off()