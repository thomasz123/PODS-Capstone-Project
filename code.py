import numpy as np
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random

random.seed(10992530)

# How I handle missing data, set threshold for min num of ratings??

#importing data from both csv files

num_data = np.genfromtxt("rmpCapstoneNum.csv")
qual_data = np.genfromtxt("rmpCapstoneQual.csv")

num_panda = pd.read_csv("rmpCapstoneNum.csv")
qual_data = pd.read_csv("rmpCaptoneQual.csv")

#clean the data


# 1: Is there evidence of pro-male gender bias?

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