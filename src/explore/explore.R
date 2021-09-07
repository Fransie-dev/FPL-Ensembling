# Libraries ---------------------------------------------------------------
library(ggplot2)
library(knitr)
library(dplyr)
library(kableExtra)
library(autoEDA)
# Separate numerical + categorical ----------------------------------------
setwd('C://Users//jd-vz//Desktop//Code//src//explore//')
df <- read.csv(file = 'C://Users//jd-vz//Desktop//Code//data//2019-20//training//cleaned_fpl.csv')
df$X <- NULL
df$player_name <- NULL
df$kickoff_time <- NULL
num_feat <- df[sapply(df, is.numeric)]
cat_feat <- df[,colnames(df)[grepl('factor|logical|character',sapply(df,class))]]


library(rattle)
library(rpart)
library(rpart.plot)
library(partykit)
library(tidyrules)

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








devtools::install_github("JaseZiv/worldfootballR")
library(worldfootballR)
understat_league_season_shots(league = "EPL", season_start_year = 2020)
