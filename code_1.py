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
all_panda = all_panda[all_panda['Num Ratings'] > 10]

all_data = all_panda.values

# 1: Is there evidence of pro-male gender bias?

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

# creating chart (maybe sd?)
# box plot??


# 2: Is there an effect of experience on the quality of teaching?
# X = experience (number of ratings)
# Y = quality of teaching (avg rating)
# linear regression
# x = all_panda[["Num Ratings"]]
# y = all_panda[["Avg Rating"]]
# model = LinearRegression()
# model.fit(x,y)


# 3: What is the relationship between average rating and average difficulty?



# 4: Do professors who teach a lot of classes in the online modality receive higher or
# lower ratings than those who don't? 

# 5: What is the relationship between the average rating and the proportion of people
# who would take the class the professor teaches again?

# 6: Do professors who are "hot" receive higher ratings thatn those who are not?

# 7: Build a regression model predicting average rating from difficulty only. 

# x = all_panda[["Avg Difficulty"]]
# y = all_panda[["Avg Rating"]]
# model7 = LinearRegression()
# model7.fit(x,y)
# slope7 = model7.coef_
# intercept7 = model7.intercept_

# plt.plot(x, y, 'o', markersize = 3)
# plt.plot(x, slope7 * x + intercept7, color = 'orange', linewidth = 3)
# plt.xlabel("Avg Difficulty")
# plt.ylabel("Avg Rating")
# plt.title("Avg Difficulty vs Avg Rating")
# plt.show()

# 8: Build a regression model predicting average rating from all available factors

df8 = all_panda.drop(["Major/Field", "University", "State"] , axis = 1).dropna()

x8 = df8.drop("Avg Rating" , axis = 1)
y8 = df8[["Avg Rating"]]

model8 = LinearRegression()
model8.fit(x8,y8)
slope8 = model8.coef_
intercept8 = model8.intercept_

plt.plot(x8, y8, 'o', markersize = 3)
plt.plot(x8, slope8 * x8 + intercept8, color = 'orange', linewidth = 3)
plt.xlabel("All Factors")
plt.ylabel("Avg Rating")
plt.title("All Factors vs Avg Rating")
plt.show()



# 9: Build a classification model that predicts whether a professor receives a "pepper"
# from average rating only. 

# 10: Build a classification model that predicts whether a professor receives a "pepper"
# from all available factors. 