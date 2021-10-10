# Libraries ---------------------------------------------------------------
library(ggplot2)
library(knitr)
library(dplyr)
library(kableExtra)
library(autoEDA)
library(rattle)
library(rpart)
library(rpart.plot)
library(partykit)
library(tidyrules)
library(papeR) 
library(xtable)
library(tidyr)

# Read data ---------------------------------------------------------------
df_f <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//data//collected_fpl.csv')
df_u <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//data//collected_us.csv')

# Check encoding ----------------------------------------------------------
head(df_f[,colnames(df_f)[grepl('factor|logical|character',sapply(df_f,class))]])
head(df_u[,colnames(df_u)[grepl('factor|logical|character',sapply(df_f,class))]])
head(df_f[sapply(df_f, is.numeric)])
head(df_u[sapply(df_u, is.numeric)])

# Number of unique entries -----------------------------------------------
sapply(names(df_f), function(var_x){print(length(unique(df_f[[var_x]])))}) 
sapply(names(df_u), function(var_x){print(length(unique(df_u[[var_x]])))})

# For easy visualization of discrete features -----------------------------
encode_factors <- function(df){
  df$player_name <- as.factor(df$player_name)
  df$position <- as.factor(df$position)
  df$kickoff_time <- as.factor(df$kickoff_time)
  leng = 3
  uniqs <- sort(sapply(df, function(x) length(unique(x))), decreasing = T)
  df_cnt <- data.frame(column = names(uniqs), nuniq = uniqs, row.names = NULL)
  print(df_cnt)
  nms <- as.character(df_cnt[df_cnt$nuniq < leng, 'column'])
  df[nms] <- lapply(df[nms], as.factor)  
  
  return(df)
}

df_f <- encode_factors(df_f)
df_u <- encode_factors(df_u)


# Distribution plots ------------------------------------------------------
plot_dist <- function(df){
  for(i in 1:ncol(df)){
    pdf(paste0(i, '.pdf'),onefile = FALSE)
    if(is.numeric(df[,i]))
    {
      if(length(unique(df[,i]))  < 30){hist(df[,i], main=colnames(df)[i], right = FALSE, breaks = seq(min(df[,i]), max(df[,i]), length.out = length(unique(df[,i]))),
                                            cex.main = 2.5, cex.lab = 1.5, cex.axis = 1.5,  col = "gray", xlab = "Values", ylab = "Frequency")}
      else{hist(df[,i], main=colnames(df)[i], right = FALSE, cex.main = 2.5, cex.lab = 1.5, cex.axis = 1.5,  col = "gray", xlab = "Values", ylab = "Frequency",breaks = 20)}
    }
    else
    {
      plot(df[,i], main=colnames(df)[i],  cex.main = 2.5, cex.names = 2,  cex.lab = 1.5, cex.axis = 1.5, xlab = "Category", ylab = "Frequency")
      
    }
    dev.off()
  }}


setwd('C://Users//jd-vz//Desktop//Code//src//explore//pdf//fpl//')
plot_dist(df_f)
setwd('C://Users//jd-vz//Desktop//Code//src//explore//pdf//understat//')
plot_dist(df_u)

# Statistical summaries ---------------------------------------------------
printx <- function(df, filename, type){
  sink(file = filename)
  print(xtable(summarize(df, type = type)))
  sink(file = NULL)}
name_date_viz <- function(df_f, df_u, ft){
  df_1 <- rbind(df_f %>% select(ft)  %>% summarize(type = 'factor') %>% select(- ' ') %>% arrange(desc(N)) %>% head(5),
                0, 
                df_f %>% select(ft)  %>% summarize(type = 'factor') %>% select(- ' ') %>% arrange(desc(N)) %>% tail(5))
  df_2 <- rbind(df_u %>% select(ft)  %>% summarize(type = 'factor') %>% select(- ' ') %>% arrange(desc(N)) %>% head(5),
                0, 
                df_u %>% select(ft)  %>% summarize(type = 'factor') %>% select(- ' ') %>% arrange(desc(N)) %>% tail(5))
  return(cbind(df_1, df_2))}


team_viz <- function(df_f, df_u){
  return(cbind(df_f %>% select(team)  %>% summarize(type = 'factor') %>% select(- ' ') %>% arrange(desc(N)),
               df_u %>% select(team)  %>% summarize(type = 'factor') %>% select(- ' ') %>% arrange(desc(N)),
               df_f %>% select(opponent_team)  %>% summarize(type = 'factor') %>% select(- ' ') %>% arrange(desc(N)),
               df_u %>% select(opponent_team)  %>% summarize(type = 'factor') %>% select(- ' ') %>% arrange(desc(N))))}

setwd('C://Users//jd-vz//Desktop//Code//src//explore//txt//fpl//')
df_f %>% printx(filename = "fpl_numeric_summary.txt", type = 'numeric')
df_f %>% select(-player_name, -kickoff_time, -team, -opponent_team) %>% printx(filename = "fpl_factor_summary.txt", type = 'factor')

setwd('C://Users//jd-vz//Desktop//Code//src//explore//txt//us//')
df_u %>% printx(filename = "us_numeric_summary.txt", type = 'numeric')
df_u %>% select(-player_name, -kickoff_time, -team, -opponent_team) %>% printx(filename = "us_factor_summary.txt", type = 'factor')

# Merge name and kickoff time ---------------------------------------------
setwd('C://Users//jd-vz//Desktop//Code//src//explore//txt//')
sink(file = 'player_names.txt'); name_date_viz(df_f, df_u, 'player_name') %>% xtable()%>% print(); sink(file = NULL)
sink(file = 'kickoff_time.txt'); name_date_viz(df_f, df_u, 'kickoff_time') %>% xtable()%>% print(); sink(file = NULL)
sink(file = 'team.txt'); team_viz(df_f, df_u) %>% xtable() %>% print(); sink(file = NULL)

# Grouped summaries -------------------------------------------------------
df_f <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//data//collected_fpl.csv')
df_u <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//data//collected_us.csv')

print_groups <- function(df, nm){ 
  for(i in c('clean_sheets', 'own_goals', 'penalties_missed', 'was_home', 'red_cards', 'was_home', 'yellow_cards', 'season')){
    filenm = paste0(paste0(nm, '_'), i, '.txt')
    sink(file = filenm)
    df[,i] <- as.factor(df[,i])
    groups <- summarize(df, type = 'numeric', group = i, test = 'wilcox.test', count = TRUE)
    df %>% select(c(groups[groups$p.value == '<0.001',1], i)) %>% summarize(group = i) %>%  xtable() %>%  print()
    df[,i] <- as.numeric(df[,i])
    sink(file = NULL)}
  for(i in c('penalties_saved', 'position', 'team', 'bonus', 'goals_scored', 'goals_conceded')){
    filenm = paste0(paste0(nm, '_'), i, '.txt')
    sink(file = filenm)
    df[,i] <- as.factor(df[,i])
    summarize(df, type = 'numeric', group = i, test = FALSE,count = TRUE) %>%  xtable() %>% print()
    sink(file = NULL)
    if(any(colnames(df) %in% c('npg', 'shots', 'key_passes'))) for(i in  c('npg', 'shots', 'key_passes')){
      filenm = paste0(paste0(nm, '_'), i, '.txt')
      sink(file = filenm)
      df[,i] <- as.factor(df[,i])
      summarize(df, type = 'numeric', group = i, test = FALSE,count = TRUE) %>%  xtable() %>% print()
      df[,i] <- as.numeric(df[,i])
      sink(file = NULL)}}}

# setwd('C://Users//jd-vz//Desktop//Code//src//explore//txt//fpl//')
# print_groups(df_f, 'fpl')
setwd('C://Users//jd-vz//Desktop//Code//src//explore//txt//us//')
print_groups(df_u, 'us')



# Updated univariate plots ------------------------------------------------
library(ggplot2); library(tidyverse); library(cowplot); 
df_f <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//data//collected_fpl.csv')
df_u <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//data//collected_us.csv')
df_f <- encode_factors(df_f)
df_u <- encode_factors(df_u)
df_f$Source <- 'Fantasy'
df_u$Source <- 'Understat'
df_f$shots <- df_f$key_passes <- df_f$xG <- df_f$xA <- df_f$npg <- df_f$npxG <- df_f$xGChain <- df_f$xGChain <- df_f$xGBuildup <-  df_f$position_stat <- NA
df_temp <- rbind(df_f,df_u)
df_temp$Source <- as.factor(df_temp$Source)
save_ggs <- function(df_temp, i, ax.tit, ax.text, wrap,sz,y_label_scl){
  if(names(df_temp[i]) %in% c('shots', 'key_passes','xG','xA','npg','npxG','xGChain','xGBuildup', 'position_stat')) df_temp <- df_temp[complete.cases(df_temp), ]
  if(is.numeric(df_temp[[i]]))
  {
    ifelse(length(unique(df_temp[[i]])) < 30 && !anyNA(df_temp[[i]]),
           ggplot(df_temp) + aes(x = df_temp[[i]]) + aes_string(names(df_temp[i])) + geom_histogram(aes(y=..density..,), colour = 'black',size = sz,
                                                                                                    breaks = seq(min(df_temp[[i]]),
                                                                                                                 max(df_temp[[i]]), 
                                                                                                                 length.out = length(unique(df_temp[[i]])))) + facet_grid(Source~.,scales = "free_x",space = "free_x")  + theme(axis.text=element_text(size=ax.tit),
                                                                                                                                                                                                                                axis.title.x=element_text(size=ax.tit),
                                                                                                                                                                                                                                axis.title.y=element_text(size=ax.tit*y_label_scl),
                                                                                                                                                                                                                                axis.text.y = element_text(size = ax.text/2),
                                                                                                                                                                                                                                strip.text = element_text(size = wrap, face = 'bold')),
           ggplot(df_temp) + aes(x = df_temp[[i]]) + aes_string(names(df_temp[i])) + geom_histogram(aes(y=..density..,), colour = 'black',size = sz) + facet_grid(Source~.,scales = "free_x",space = "free_x")+ theme(axis.text=element_text(size=ax.tit),
                                                                                                                                                                                                                      axis.title.x=element_text(size=ax.tit),
                                                                                                                                                                                                                      axis.title.y=element_text(size=ax.tit*y_label_scl),
                                                                                                                                                                                                                      axis.text.x = element_text(size = ax.text),
                                                                                                                                                                                                                      axis.text.y = element_text(size = ax.text/2),
                                                                                                                                                                                                                      strip.text = element_text(size = wrap, face = 'bold')))
    
  }
  
  else {if(length(unique(df_temp[[i]])) < 20) ggplot(df_temp) + aes(x = df_temp[[i]]) + aes_string(names(df_temp[i])) +  geom_bar(colour = 'black',size = sz)+ facet_grid(Source~.,scales = "free_x",space = "free_x") + theme(axis.text=element_text(size=ax.tit),
                                                                                                                                                                                                                               axis.title.x=element_text(size=ax.tit),
                                                                                                                                                                                                                               axis.title.y=element_text(size=ax.tit*y_label_scl),
                                                                                                                                                                                                                               axis.text.x = element_text(size = ax.text),
                                                                                                                                                                                                                               axis.text.y = element_text(size = ax.text/2),
                                                                                                                                                                                                                               strip.text = element_text(size = wrap, face = 'bold'))
    else ggplot(df_temp) + aes(x = df_temp[[i]]) + aes_string(names(df_temp[i])) +  geom_bar(colour = 'black',size = sz)+ facet_grid(Source~.,scales = "free_x",space = "free_x") + theme(axis.text=element_text(size=ax.tit),
                                                                                                                                                                                          axis.title.x=element_text(size=ax.tit),
                                                                                                                                                                                          axis.title.y=element_text(size=ax.tit*y_label_scl),
                                                                                                                                                                                          axis.text.x=element_blank(),                                                                                                                                                                                                axis.text.y = element_text(size = ax.text/2),
                                                                                                                                                                                          strip.text = element_text(size = wrap, face = 'bold'))}
  
  
  
  
  ggsave(filename = paste0(i, '.pdf'), width = 15, height = 15)
}
setwd('C://Users//jd-vz//Desktop//Code//src//explore//pdf//fpl//')
for(i in 1:(ncol(df_temp))) save_ggs(df_temp, i, ax.tit = 75, ax.text = 50, wrap = 50, sz = 3.5,y_label_scl = 0.4)

library(dplyr)
# Non-duplicated entries --------------------------------------------------
df_f <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//data//collected_fpl.csv')
df_u <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//data//collected_us.csv')
df_u$shots <- df_u$key_passes <- df_u$xG <- df_u$xA <- df_u$npg <- df_u$npxG <- df_u$xGChain <- df_u$xGChain <- df_u$xGBuildup <- NA
dup <- anti_join(df_u, df_f)
unique(dup$player_name) # Instances in understat data not in fpl data 
dup <- anti_join(df_f, df_u) # Instances in fpl data not in us data 
length(unique(dup$player_name)) 
unique(dup$player_name)
dup <- anti_join(df_f[df_f$minutes > 0, ], df_u[df_u$minutes > 0, ]) # Instances in fpl data not in us data with non-zero min
a <- df_f[df_f$player_name %in% unique(dup$player_name),]
a <- a[a$minutes > 0,] # Three players logged
a
nrow(a[a$player_name == 'Vitor Ferreira',])
nrow(a[a$player_name == 'Adrián San Miguel del Castillo',])
nrow(a[a$player_name == 'João Pedro Junqueira de Jesus',])

max(a[a$player_name == 'Vitor Ferreira','transfers_balance'])
max(a[a$player_name == 'Adrián San Miguel del Castillo','transfers_balance'])
max(a[a$player_name == 'João Pedro Junqueira de Jesus','transfers_balance'])

mean(a[a$player_name == 'Vitor Ferreira','total_points'])
mean(a[a$player_name == 'Adrián San Miguel del Castillo','total_points'])
mean(a[a$player_name == 'João Pedro Junqueira de Jesus','total_points'])


summary(a[a$player_name == 'Adrián San Miguel del Castillo',c('total_points', 'minutes')]) %>% kable('latex')

summarize(a[a$player_name == 'Adrián San Miguel del Castillo',c('total_points', 'minutes')]) %>% xtable() %>% print()
a <- droplevels.data.frame(a)
summarize(a[c('total_points', 'minutes', 'player_name')], group = 'player_name', test = FALSE) %>% xtable() %>% print()
a <- droplevels.data.frame(a)

nrow(df_f[df_f$total_points == -1,])/nrow(df_f)
nrow(df_u[df_u$total_points == -1,])/nrow(df_u)

nrow(df_f[df_f$total_points == 0,])/nrow(df_f)
nrow(df_u[df_u$total_points == 0,])/nrow(df_u)


a[a$player_name == 'Vitor Ferreira',]
a[a$player_name == 'João Pedro Junqueira de Jesus',]
# Initial distributions ---------------------------------------------------
for(i in c('clean_sheets', 'own_goals', 'penalties_missed', 'was_home', 'red_cards', 'was_home', 'yellow_cards', 'season')) autoEDA(df_u, i, sampleRate = 1,
                                                                                                                                    outcomeType = "automatic", maxUniques = 15, maxLevels = 25,
                                                                                                                                    removeConstant = FALSE, removeZeroSpread = FALSE,
                                                                                                                                    removeMajorityMissing = FALSE, imputeMissing = FALSE, clipOutliers = FALSE,
                                                                                                                                    minLevelPercentage = 0.025, predictivePower = TRUE,
                                                                                                                                    outlierMethod = "tukey", lowPercentile = 0.01, upPercentile = 0.99,
                                                                                                                                    plotCategorical = "stackedBar", plotContinuous = "histogram",
                                                                                                                                    outputPath = 'C://Users//jd-vz//Desktop//Code//src//explore//pdf//',
                                                                                                                                    filename = i,
                                                                                                                                    verbose = TRUE)


# Pairs plot of ten highest correlated features ---------------------------
library(ggplot2)
library(GGally)
top_10 <- c('total_points', 'position', 'bps', 'bonus', 'influence')
df_u <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//data//collected_us.csv')
df_u <- df_u[top_10]
ggpairs(df_u, mapping = aes(colour = position, alpha = 0.6), title="Pairs plot: Unscaled, clustered dataset with Gender and Cluster included") + theme_grey(base_size = 10)



data_pp <- data_cc
data_pp$Gender <- NULL
cols <- 1:ncol(data_pp)
scl_select <- "uniminmax"
ggparcoord(data = data_pp, columns = cols[-ncol(data_pp)], groupColumn = "cluster", title = "Parallel Coord. Plot of Abalone Data Grouped by Cluster",
           scale = scl_select, alphaLines = 0.1, order = "allClass") +  scale_colour_discrete()
ggparcoord(data = data_pp, columns = cols[-c(c(ncol(data_pp)-1):ncol(data_pp))], groupColumn = "Rings", title = "Parallel Coord. Plot of Abalone Data Grouped by Rings",
           scale = scl_select, alphaLines = 0.1, order = "allClass") +  scale_color_discrete()








p1 <- ggplot(df_f,aes(cut,price,fill=color)) + geom_bar(stat = "identity", na.rm=TRUE)



ggplot(df, aes(fill=condition, y=value, x=specie)) + 
  geom_bar(position="dodge", stat="identity")




summarize(df_u, type = 'numeric', group = i, test = 'wilcox.test', count = TRUE)





papeR::prettify(summary(data.frame(df_u)))

prettify(df_f,labels = 'total_points', sep = ": ", extra.column = FALSE,
         smallest.pval = 0.001, digits = NULL, scientific = FALSE,
         signif.stars = getOption("show.signif.stars"))

var_x <- 'bps'
ggplot(df_temp) + aes(x = var_x, fill = Source) + aes_string(var_x) + geom_histogram(aes(y=..density..),alpha = 0.2)


var_x <- 'player_name'
ggplot(df_temp) + aes(x = var_x, fill = Source) + aes_string(var_x) + geom_bar(alpha = 0.2)

autoEDA(df_temp, y = 'Source', IDFeats = NULL, sampleRate = 1,
        outcomeType = "automatic", maxUniques = 15, maxLevels = 25,
        removeConstant = FALSE, removeZeroSpread = FALSE,
        removeMajorityMissing = FALSE, imputeMissing = FALSE, clipOutliers = FALSE,
        minLevelPercentage = 0.025, predictivePower = TRUE,
        outlierMethod = "tukey", lowPercentile = 0.01, upPercentile = 0.99,
        plotCategorical = "stackedBar", plotContinuous = "density", bins = 20,
        rotateLabels = FALSE, colorTheme = 1, theme = 2, color = "#26A69A",
        transparency = 0.005, outputPath = 'C://Users//jd-vz//Desktop//Code//src//explore//', filename = "all_positions",
        verbose = TRUE)



nrow(df_u[df_u$minutes == 0,])

nrow(df_f) - 25860

nrow(df_f[df_f$minutes == 0,])/nrow(df_f) * 100

nrow(df_f[df_f$minutes == 0,])
nrow(df_u[df_u$minutes == 0,])

length(unique(df_f[df_f$minutes > 0, 'player_name']))
length(unique(df_u[df_u$minutes > 0, 'player_name']))


max(df_f[df_f$minutes > 0, 'total_points'])
max(df_u[df_u$minutes > 0, 'total_points'])









# Group by position -------------------------------------------------------
df_temp <- df_f
check_me <- summarize(df_temp, type = "numeric", group = 'position', test = FALSE)
View(check_me)


# Group by clean sheets ---------------------------------------------------
df_temp <- df
df_temp$clean_sheets <- as.factor(df_temp$clean_sheets)
check_me <- summarize(df_temp, type = "numeric", group = 'clean_sheets', test = TRUE)
View(check_me)


# Group by penalties missed -----------------------------------------------
df_temp <- df
df_temp$penalties_missed <- as.factor(df_temp$penalties_missed)
check_me <- summarize(df_temp, type = "numeric", group = 'penalties_missed', test = TRUE)
View(check_me)



xtable(summarize(df, type = "factor", variables = "clean_sheets"))
xtable(summarize(df, type = "factor", variables = "penalties_missed"))
xtable(summarize(df, type = "factor", variables = "red_cards"))
xtable(summarize(df, type = "factor", variables = "yellow_cards"))
xtable(summarize(df, type = "factor", variables = "was_home"))


a <- summarize(df, type = "numeric", group = "team", test = FALSE)




# According to all positions ----------------------------------------------
autoEDA(df_temp, IDFeats = NULL, sampleRate = 1,
        outcomeType = "automatic", maxUniques = 15, maxLevels = 25,
        removeConstant = FALSE, removeZeroSpread = FALSE,
        removeMajorityMissing = FALSE, imputeMissing = FALSE, clipOutliers = FALSE,
        minLevelPercentage = 0.025, predictivePower = TRUE,
        outlierMethod = "tukey", lowPercentile = 0.01, upPercentile = 0.99,
        plotCategorical = "stackedBar", plotContinuous = "density", bins = 20,
        rotateLabels = FALSE, colorTheme = 1, theme = 2, color = "#26A69A",
        transparency = 0.005, outputPath = 'C://Users//jd-vz//Desktop//Code//src//explore//', filename = "all_positions",
        verbose = TRUE)













fit <- rpart(total_points ~., data = df,
             parms = list(split = "information"),
             control = rpart.control(cp = 0,
                                     xval = 0,
                                     minsplit = 5,
                                     maxdepth = 5,
                                     minbucket = 10))

summary(fit)

fancyRpartPlot(fit,
               type = 5,
               main = "Unpruned CART",
               sub = NULL)

fit_party <- as.party.rpart(fit)
fit_party

rules_tidy <- tidyRules(fit)
rules_tidy



check <- data.frame(ID = rules_tidy$id, 
                    IF = rules_tidy$LHS,
                    THEN = rules_tidy$RHS,
                    Support = rules_tidy$support,
                    # Coverage = paste0(round(rules_tidy$support/nrow(train_smote)*100,3), "%"),
                    Coverage = rules_tidy$support/nrow(df),
                    Accuracy = rules_tidy$confidence)

# Univariate analysis -----------------------------------------------------

# According to all positions ----------------------------------------------
autoEDA(df_temp, y = 'position', IDFeats = NULL, sampleRate = 1,
        outcomeType = "automatic", maxUniques = 15, maxLevels = 25,
        removeConstant = FALSE, removeZeroSpread = FALSE,
        removeMajorityMissing = FALSE, imputeMissing = FALSE, clipOutliers = FALSE,
        minLevelPercentage = 0.025, predictivePower = TRUE,
        outlierMethod = "tukey", lowPercentile = 0.01, upPercentile = 0.99,
        plotCategorical = "stackedBar", plotContinuous = "density", bins = 20,
        rotateLabels = FALSE, colorTheme = 1, theme = 2, color = "#26A69A",
        transparency = 0.005, outputPath = 'C://Users//jd-vz//Desktop//Code//src//explore//', filename = "all_positions",
        verbose = TRUE)


# According to GK ---------------------------------------------------------
df_temp <- df
df_temp$position <- recode_factor(df_temp$position, FWD = 'other', DEF = 'other', MID = 'other')
autoEDA(df_temp, y = 'position', IDFeats = NULL, sampleRate = 1,
        outcomeType = "automatic", maxUniques = 15, maxLevels = 25,
        removeConstant = FALSE, removeZeroSpread = FALSE,
        removeMajorityMissing = FALSE, imputeMissing = FALSE, clipOutliers = FALSE,
        minLevelPercentage = 0.025, predictivePower = TRUE,
        outlierMethod = "tukey", lowPercentile = 0.01, upPercentile = 0.99,
        plotCategorical = "stackedBar", plotContinuous = "density", bins = 20,
        rotateLabels = FALSE, colorTheme = 1, theme = 2, color = "#26A69A",
        transparency = 0.005, outputPath = 'C://Users//jd-vz//Desktop//Code//src//explore//', filename = "GK",
        verbose = TRUE)


# According to FWD --------------------------------------------------------
df_temp <- df
df_temp$position <- recode_factor(df_temp$position, GK = 'other', DEF = 'other', MID = 'other')
autoEDA(df_temp, y = 'position', IDFeats = NULL, sampleRate = 1,
        outcomeType = "automatic", maxUniques = 15, maxLevels = 25,
        removeConstant = FALSE, removeZeroSpread = FALSE,
        removeMajorityMissing = FALSE, imputeMissing = FALSE, clipOutliers = FALSE,
        minLevelPercentage = 0.025, predictivePower = TRUE,
        outlierMethod = "tukey", lowPercentile = 0.01, upPercentile = 0.99,
        plotCategorical = "stackedBar", plotContinuous = "density", bins = 20,
        rotateLabels = FALSE, colorTheme = 1, theme = 2, color = "#26A69A",
        transparency = 0.005, outputPath = 'C://Users//jd-vz//Desktop//Code//src//explore//', filename = "FWD",
        verbose = TRUE)

# According to DEF --------------------------------------------------------
df_temp <- df
df_temp$position <- recode_factor(df_temp$position, FWD = 'other', GK = 'other', MID = 'other')
autoEDA(df_temp, y = 'position', IDFeats = NULL, sampleRate = 1,
        outcomeType = "automatic", maxUniques = 15, maxLevels = 25,
        removeConstant = FALSE, removeZeroSpread = FALSE,
        removeMajorityMissing = FALSE, imputeMissing = FALSE, clipOutliers = FALSE,
        minLevelPercentage = 0.025, predictivePower = TRUE,
        outlierMethod = "tukey", lowPercentile = 0.01, upPercentile = 0.99,
        plotCategorical = "stackedBar", plotContinuous = "density", bins = 20,
        rotateLabels = FALSE, colorTheme = 1, theme = 2, color = "#26A69A",
        transparency = 0.005, outputPath = 'C://Users//jd-vz//Desktop//Code//src//explore//', filename = "DEF",
        verbose = TRUE)

# According to MID --------------------------------------------------------
df_temp <- df
df_temp$position <- recode_factor(df_temp$position, FWD = 'other', DEF = 'other', GK = 'other')
autoEDA(df_temp, y = 'position', IDFeats = NULL, sampleRate = 1,
        outcomeType = "automatic", maxUniques = 15, maxLevels = 25,
        removeConstant = FALSE, removeZeroSpread = FALSE,
        removeMajorityMissing = FALSE, imputeMissing = FALSE, clipOutliers = FALSE,
        minLevelPercentage = 0.025, predictivePower = TRUE,
        outlierMethod = "tukey", lowPercentile = 0.01, upPercentile = 0.99,
        plotCategorical = "stackedBar", plotContinuous = "density", bins = 20,
        rotateLabels = FALSE, colorTheme = 1, theme = 2, color = "#26A69A",
        transparency = 0.005, outputPath = 'C://Users//jd-vz//Desktop//Code//src//explore//', filename = "MID",
        verbose = TRUE)



all_us[c(1, 4095, 932, 120, 6250),c('goals','xG', 'xA', 'date','player_name', 'h_team', 'a_team')] %>%  kable( "latex", longtable = T, booktabs = T)

# Second gameweek table
gw[c('element','fixture','opponent_team','total_points','was_home', 'kickoff_time')]%>% head(5) %>%  kable( "latex", longtable = T, booktabs = T)

# All features with respect to position -----------------------------------
autoEDA(df, y = 'position', IDFeats = NULL, sampleRate = 1,
        outcomeType = "automatic", maxUniques = 15, maxLevels = 25,
        removeConstant = FALSE, removeZeroSpread = FALSE,
        removeMajorityMissing = FALSE, imputeMissing = FALSE, clipOutliers = FALSE,
        minLevelPercentage = 0.025, predictivePower = TRUE,
        outlierMethod = "tukey", lowPercentile = 0.01, upPercentile = 0.99,
        plotCategorical = "stackedBar", plotContinuous = "density", bins = 20,
        rotateLabels = FALSE, colorTheme = 1, theme = 2, color = "#26A69A",
        transparency = 1, outputPath = 'C://Users//jd-vz//Desktop//Code//src//explore//', filename = "ExploratoryPlots",
        verbose = TRUE)


table <- dataOverview(num_feat, outlierMethod = "tukey", lowPercentile = 0.01,
                      upPercentile = 0.99, minLevelPercentage = 0.025)


kable(table, "latex", longtable = T, booktabs = T) %>%
  kable_styling(latex_options = c("repeat_header"), font_size = 7) %>%
  landscape()

gw[c('element','fixture','opponent_team','total_points','was_home')]%>% head(5) %>%  kable( "latex", longtable = T, booktabs = T)



df %>%
  ggplot(aes(x = value, y = total_points, color = position)) +
  geom_line() +
  gghighlight::gghighlight(position == "FWD" | position == "GK") +
  labs(
    title = "Twenty most commonly given names",
    y = "proportion of babies",
    subtitle = "Elizabeth stable over time; Mary decreasing"
  )


data_scaled <- scale(num_feat)
set.seed(0)
sample_size <- 0.01*nrow(data_scaled) 
ss <- sample(nrow(data_scaled), sample_size) # Random rows
factoextra::fviz_dist(factoextra::get_dist(data_scaled, method = "euclidean"),
                      order = T, show_labels = F, gradient = list(low = "red",
                                                                  mid = "white",
                                                                  high = "blue")) +
  ggplot2::labs(title = paste("ODI Scaled Abalone Data; Sample Percentage: ", 
                              paste0(100*0.01, "%")))

# ggplot(data = df, aes(x = value ,y = total_points)) +
#         geom_point(alpha = 0.5, aes(color=position))


merged_gw[c('element','fixture','opponent_team','total_points','was_home')]%>% head(5) %>%  kable( "latex", longtable = T, booktabs = T)


