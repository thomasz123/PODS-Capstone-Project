import numpy as np
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random

random.seed(10992530)

#importing data from both csv files
num_panda = pd.read_csv("rmpCapstoneNum.csv", names = ["Avg Rating", "Avg Difficulty", "Num Ratings", "Pepper", "Would Retake", "Online", "Male", "Female"])
qual_panda = pd.read_csv("rmpCapstoneQual.csv",names = ["Major/Field", "University", "State"])

#combine two data files
all_panda = num_panda.join(qual_panda)

# filter by num ratings > 10
all_panda = all_panda[all_panda['Num Ratings'] > 5]

all_data = all_panda.values

# 1: Is there evidence of pro-male gender bias?

# significance test
male_ratings = all_panda[all_panda["Male"] == 1 ][["Avg Rating", "Male"]]
female_ratings = all_panda[all_panda["Female"] == 1][["Avg Rating", "Female"]]
print("Average male ratings: ", male_ratings["Avg Rating"].mean())
print("Average female ratings: ", female_ratings["Avg Rating"].mean())
print("Variance of male ratings: ", male_ratings["Avg Rating"].var())
print("Variance of male ratings: ", female_ratings["Avg Rating"].var())

# We will perform a significance test. We will use the independent samples t-test
# because the ratings of each professor are independent and the variances are similar.
tstat, pvalue = stats.ttest_ind(male_ratings["Avg Rating"], female_ratings["Avg Rating"])
print("p-value:", pvalue)


# perform significance testing 

# 2: Is there an effect of experience on the quality of teaching?

# 3: What is the relationship between average rating and average difficulty?

# 4: Do professors who teach a lot of classes in the online modality receive higher or
# lower ratings than those who don't? 

# 5: What is the relationship between the average rating and the proportion of people
# who would take the class the professor teaches again?

# 6: Do professors who are "hot" receive higher ratings thatn those who are not?

# 7: Build a regression model predicting average rating from difficulty only. 

# 8: Build a regression model predicting average rating from all available factors

# 9: Build a classification model that predicts whether a professor receives a "pepper"
# from average rating only. 

# 10: Build a classification model that predicts whether a professor receives a "pepper"
# from all available factors. 