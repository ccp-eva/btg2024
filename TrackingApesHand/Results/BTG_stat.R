library(tidyverse)
library(ggplot2)
library(gplots)
library(lme4)
library(pgirmess)

direction_change = read_csv("direction_change.csv")

df_anticipation = direction_change %>%
  mutate(correct_pos = ifelse(loc == pos_2,"Correct","Others")) %>% 
  filter(trial_stage=="RSI_2" & loc!= pos_1)
  

plotmeans(df_anticipation$change_bin~df_anticipation$correct_pos)


ggplot(df_anticipation, aes(x=correct_pos, y=change_bin, color = correct_pos))+
  geom_point(stat = "summary")+
  stat_summary(fun.data=mean_cl_boot, geom="errorbar", width=0.2) +
  theme(legend.position ="none")+
  facet_wrap(~subject,nrow=1)+
  labs(x= "Location",y="Proportion of direction change during the 2nd delay")


model = glm(change_bin~correct_pos, df_anticipation, family = "binomial")
summary(model)
