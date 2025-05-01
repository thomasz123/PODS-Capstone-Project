import numpy as np
import pandas as pd 
from scipy import stats
from scipy.special import expit
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, r2_score, mean_squared_error
import random

random.seed(10992530)

#importing data from both csv files
num_panda = pd.read_csv("rmpCapstoneNum.csv", names = ["Avg Rating", "Avg Difficulty", "Num Ratings", "Pepper", "Would Retake", "Online", "Male", "Female"])
qual_panda = pd.read_csv("rmpCapstoneQual.csv",names = ["Major/Field", "University", "State"])

#combine two data files
all_panda = num_panda.join(qual_panda)

# filter by num ratings > 10
all_panda = all_panda[all_panda['Num Ratings'] > 10]

# 1: Is there evidence of pro-male gender bias?
print ("--------------- Question 1 ---------------")
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

fig = plt.figure()
# ax = fig.add_axes([0, 1])
plt.boxplot([male_ratings["Avg Rating"], female_ratings["Avg Rating"]])
plt.xticks([1, 2], ['Male Ratings', 'Female Ratings'])
# plt.xlabel("Experience")
plt.ylabel("Avg Rating")
plt.title("Male Ratings vs Female Ratings")
plt.show() 

# 2: Is there an effect of experience on the quality of teaching?
print ("--------------- Question 2 ---------------")

q2 = all_panda[all_panda['Num Ratings'] > 25].dropna()

experience = q2["Num Ratings"]
quality = q2["Avg Rating"]

tstat2, pvalue2 = stats.ttest_rel(experience, quality)
print("p-value:", pvalue2)

print("correlation coef: " , experience.corr(quality, method = 'spearman'))

plt.figure()
plt.scatter(experience, quality, alpha = 0.5)
plt.xlabel("Experience")
plt.ylabel("Quality")
plt.title("Experience vs quality")
plt.show() 

# 3: What is the relationship between avg rating and avg difficulty?
print ("--------------- Question 3 ---------------")
q3 = all_panda[["Avg Rating", "Avg Difficulty"]].dropna()
avg_rating = q3[["Avg Rating"]]
avg_diff = q3[["Avg Difficulty"]]

model2 = LinearRegression()
model2.fit(avg_rating.values,avg_diff.values)
slope2 = model2.coef_
intercept2 = model2.intercept_

plt.figure()
plt.scatter(avg_rating, avg_diff,alpha = 0.5)
plt.plot(avg_rating, slope2 * avg_rating + intercept2, color = 'black', linewidth = 3)
plt.xlabel("Avg Difficulty")
plt.ylabel("Avg Rating")
plt.title("Avg Difficulty vs Avg Rating")
plt.show()

tstat3, pvalue3 = stats.ttest_rel(avg_rating, avg_diff)
print("p-value:", pvalue3)

pearson_coef = q3["Avg Rating"].corr(q3["Avg Difficulty"])
spearman_coef = q3["Avg Rating"].corr(q3["Avg Difficulty"], method = "spearman")

print("pearson: ", pearson_coef)
print("spearman: ", spearman_coef)

# 4: Do professors who teach a lot of classes in the online modality receive higher or
# lower ratings than those who don't? 

print ("--------------- Question 4 ---------------")
all_panda["Proportion Online"] = all_panda["Online"] / all_panda["Num Ratings"]

q4 = all_panda[all_panda["Proportion Online"] > 0]
median = q4["Proportion Online"].median()

high = q4[q4["Proportion Online"] >= median]
low = q4[q4["Proportion Online"] < median]

print(len(high))
print(len(low))

fig = plt.figure()
# ax = fig.add_axes([0, 1])
plt.boxplot([low["Avg Rating"], high["Avg Rating"]])
plt.xticks([1, 2], ['Less Online Classes', 'More Online Classes'])
# plt.xlabel("Experience")
plt.ylabel("Avg Rating")
plt.title("Online Classes vs Avg Rating")
plt.show() 

print("More Online Avg: ", high["Avg Rating"].mean())
print("Less Online Avg: ", low["Avg Rating"].mean())
print("Variance of More Online Avg: ", high["Avg Rating"].var())
print("Variance of Less Online Avg: ", low["Avg Rating"].var())

tstat4, pvalue4 = stats.ttest_rel(all_panda["Proportion Online"], all_panda["Avg Rating"])
print("p-value:", pvalue4)

# 5: What is the relationship between the average rating and the proportion of people
# who would take the class the professor teaches again?

print ("--------------- Question 5 ---------------")

q5 = all_panda[["Avg Rating", "Would Retake"]].dropna()

retake = q5[["Would Retake"]]
avg_rating5 = q5[["Avg Rating"]]

model5 = LinearRegression()
model5.fit(retake, avg_rating5)
slope5 = model5.coef_[0]
intercept5 = model5.intercept_

plt.figure()
plt.scatter(retake, avg_rating5 ,alpha = 0.5)
plt.plot(retake, slope5 * retake + intercept5, color = 'black', linewidth = 3)
plt.xlabel("Retake Proportion")
plt.ylabel("Avg Rating")
plt.title("Would Retake Proportion vs Avg Rating")
plt.show()

pearson_coef = q5["Would Retake"].corr(q5["Avg Rating"])

print("coef: ", pearson_coef)

tstat, pvalue = stats.ttest_rel(q5["Avg Rating"], q5["Would Retake"])
print("p-value:", pvalue)

# 6: Do professors who are "hot" receive higher ratings than those who are not?

print ("--------------- Question 6 ---------------")
hot = all_panda[all_panda["Pepper"] == 1]
not_hot = all_panda[all_panda["Pepper"] == 0]
print("Average hot ratings: ", hot["Avg Rating"].mean())
print("Average not hot ratings: ", not_hot["Avg Rating"].mean())
print("Variance of hot ratings: ", hot["Avg Rating"].var())
print("Variance of not hot ratings: ", not_hot["Avg Rating"].var())

# We will perform a significance test. We will use the independent samples t-test
# because the ratings of each professor are independent and the variances are similar.
tstat, pvalue = stats.ttest_ind(hot["Avg Rating"], not_hot["Avg Rating"])
print("p-value:", pvalue)

fig = plt.figure()
# ax = fig.add_axes([0, 1])
plt.boxplot([hot["Avg Rating"], not_hot["Avg Rating"]])
plt.xticks([1, 2], ['Hot Ratings', 'Not Hot Ratings'])
# plt.xlabel("Experience")
plt.ylabel("Avg Rating")
plt.title("Hot Professor Ratings vs Not Hot Professor Ratings")
plt.show() 

# 7: Build a regression model predicting average rating from difficulty only. 

print ("--------------- Question 7 ---------------")
x = all_panda[["Avg Difficulty"]]
y = all_panda[["Avg Rating"]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
model7 = LinearRegression()
model7.fit(X_train,y_train)
slope7 = model7.coef_
intercept7 = model7.intercept_

prediction = model7.predict(X_test)
r2  = r2_score(y_test, prediction)
rmse = (mean_squared_error(y_test, prediction)) ** 0.5
print("r2: ", r2)
print("rmse: ", rmse)

plt.plot(x, y, 'o', markersize = 3)
plt.plot(x, slope7 * x + intercept7, color = 'black', linewidth = 3)
plt.xlabel("Avg Difficulty")
plt.ylabel("Avg Rating")
plt.title("Avg Difficulty vs Avg Rating")
plt.show()

# 8: Build a regression model predicting average rating from all available factors

print ("--------------- Question 8 ---------------")

q8 = all_panda.drop(["Major/Field", "University", "State"], axis = 1).dropna()
predictors = q8.drop("Avg Rating", axis = 1).to_numpy()
yOutcomes = q8["Avg Rating"].to_numpy()

r = np.corrcoef(predictors,rowvar=False)
plt.imshow(r) 
plt.colorbar()
plt.show()

# There is some correlation between variables so we must perform a PCA.

# Z-score the data:
zscoredData = stats.zscore(predictors)

# Initialize PCA object and fit to our data:
pca = PCA().fit(zscoredData)

# Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_

# Loadings (eigenvectors): Weights per factor in terms of the original data.
loadings = pca.components_*-1

# Rotated Data - simply the transformed data:
origDataNewCoordinates = pca.fit_transform(zscoredData)*-1

# Scree plot:
plt.bar(np.linspace(1,8,8),eigVals)
plt.title('Scree plot')
for i, value in enumerate(eigVals):
    plt.text(i + 1, value, str(round(value,2)), ha='center', va='bottom')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

X = np.column_stack((origDataNewCoordinates[:,0],origDataNewCoordinates[:,1], origDataNewCoordinates[:,2], origDataNewCoordinates[:,3]))

X_train, X_test, y_train, y_test = train_test_split(X, yOutcomes, test_size = 0.2, random_state=42)
model8 = LinearRegression()
model8.fit(X_train,y_train)
slope8 = model8.coef_
intercept8 = model8.intercept_

prediction = model8.predict(X_test)
r2  = r2_score(y_test, prediction)
rmse = (mean_squared_error(y_test, prediction)) ** 0.5
print("r2: ", r2)
print("rmse: ", rmse)


# 9: Build a classification model that predicts whether a professor receives a "pepper"
# from average rating only. 

print ("--------------- Question 9 ---------------")

q9 = all_panda[["Avg Rating", "Pepper"]].dropna()
pepper9 = q9[["Pepper"]]
avg_rating9 = q9[["Avg Rating"]]

print(pepper9.value_counts())

X_train, X_test, y_train, y_test = train_test_split(avg_rating9, pepper9, test_size = 0.2, random_state = 42)
model9 = LogisticRegression()
model9.fit(X_train, y_train)

x1 = np.linspace(1,5,500)
y1 = x1 * model9.coef_ + model9.intercept_
sigmoid = expit(y1)

# plot data: 
plt.scatter(avg_rating9, pepper9, alpha = 0.2)
plt.xlabel('Average Rating')
plt.ylabel('Pepper')
plt.plot(x1,sigmoid.ravel(),color='red',linewidth=3)
plt.yticks(np.array([0,1]))
plt.show()

#define metrics
y_pred_proba = model9.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc=4)
plt.show()


# 10: Build a classification model that predicts whether a professor receives a "pepper"
# from all available factors. 

print ("--------------- Question 10 ---------------")

q10 = all_panda.drop(["Major/Field", "University", "State"], axis = 1).dropna()
predictors = q10.drop("Pepper", axis = 1).to_numpy()
yOutcomes = q10.Pepper.to_numpy()

r = np.corrcoef(predictors,rowvar=False)
plt.imshow(r) 
plt.colorbar()
plt.show()

# There is some correlation between variables so we must perform a PCA.

# Z-score the data:
zscoredData = stats.zscore(predictors)

# Initialize PCA object and fit to our data:
pca = PCA().fit(zscoredData)

# Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_

# Loadings (eigenvectors): Weights per factor in terms of the original data.
loadings = pca.components_*-1

# Rotated Data - simply the transformed data:
origDataNewCoordinates = pca.fit_transform(zscoredData)*-1

# Scree plot:
plt.bar(np.linspace(1,8,8),eigVals)
plt.title('Scree plot')
for i, value in enumerate(eigVals):
    plt.text(i + 1, value, str(round(value,2)), ha='center', va='bottom')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

fig, ax = plt.subplots(2,2)
ax[0,0].bar(np.linspace(1,8,8),loadings[0,:]) 
ax[0,0].set_title('Factor 1')
ax[0,1].bar(np.linspace(1,8,8),loadings[1,:]) 
ax[0,1].set_title('Factor 2')
ax[1,0].bar(np.linspace(1,8,8),loadings[2,:]) 
ax[1,0].set_title('Factor 3')
ax[1,1].bar(np.linspace(1,8,8),loadings[3,:]) 
ax[1,1].set_title('Factor 4')
plt.show()

X = np.column_stack((origDataNewCoordinates[:,0],origDataNewCoordinates[:,1], origDataNewCoordinates[:,2], origDataNewCoordinates[:,3]))

X_train, X_test, y_train, y_test = train_test_split(X, yOutcomes, test_size = 0.2, random_state = 42)
model10 = LogisticRegression()
model10.fit(X_train, y_train)

#define metrics
y_pred_proba = model10.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc=4)
plt.show()

print ("--------------- Extra Credit ---------------")
# Extra credit: Which states have the highest professor ratings? 
state_rank = (all_panda.groupby('State', as_index=False)['Avg Rating'].mean().rename(\
columns={'Avg Rating':'mean_rating'}).sort_values('mean_rating', ascending=False))

#cLook at the top 5 states :
print(state_rank.head(5))

# Which majors/fields have the highest professor ratings? 
major_counts = (all_panda.groupby('Major/Field').size().reset_index(name='prof_count'))

# Filter to majors/fields with at least 10 professors
valid_majors = major_counts.loc[major_counts['prof_count'] >= 10, 'Major/Field']

major_rank = (
    all_panda[all_panda['Major/Field'].isin(valid_majors)]
    .groupby('Major/Field', as_index=False)
    .agg(mean_rating=('Avg Rating', 'mean'))
    .sort_values('mean_rating', ascending=False)
)

# Look at top 5 majors/fields
print(major_rank.head(5))