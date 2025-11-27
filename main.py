import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# read the data
teams = pd.read_csv("teams.csv")
teams = teams[["team", "country", "year",
               "athletes", "age", "prev_medals", "medals"]]

# determine if any correlation exists
teams_numeric = teams.select_dtypes(include="number")
print("Correlation:\n", teams_numeric.corr()["medals"])

print("\nStarting Plot:\n", teams)

# explore data
# plot different variables to number of medals earned
# sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)
# sns.lmplot(x="age", y="medals", data=teams, fit_reg=True, ci=None)
# teams.plot.hist(y="medals")
plt.show()

# clean the data (remove missing values)
teams[teams.isnull().any(axis=1)]
teams = teams.dropna()
print("\nCleaned Plot (Removed missing values): \n", teams)

# split the data
# train model on training set (~80%)
train = teams[teams["year"] < 2012].copy()
# validate model with testing set (~20%)
test = teams[teams["year"] >= 2012].copy()
print("\nTrain shape (~80%):", train.shape)
print("Test shape (~20%):", test.shape)

# train the model and evaluate with mean absolute error estimate
# use linear regression model to use predictors to predict the target
reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
target = "medals"

# fit the model (find line of best fit to find patterns as close to dataset)
reg.fit(train[predictors], train["medals"])
LinearRegression()

# make predictions using current model
predictions = reg.predict(test[predictors])  # "medals" aren't rounded/negative
test["predictions"] = predictions
print("\nOriginal Predictions:\n", test)

# rescale predictions
test["predictions"] = predictions.round()
test.loc[test["predictions"] < 0, "predictions"] = 0
print("\nRescaled Predictions:\n", test)

# measure error (mean absolute error)
error = mean_absolute_error(test["medals"], test["predictions"])
print("\nMean Absolute Error:", error)
print(teams.describe()["medals"])

# compare predictions between different countires
print("\nTeam USA Predictions:\n", test[test["team"] == "USA"])
print("\nTeam India Predictions:\n", test[test["team"] == "IND"])

errors = (test["medals"] - test["predictions"]).abs()
print("\nErrors by country:\n", errors)

# check how many medals model was off by for each country
error_by_team = errors.groupby(test["team"]).mean()
print("\nErrors by team:\n", error_by_team)

# check how many medals country earned by average
medals_by_team = test["medals"].groupby(test["team"]).mean()

# clean and plot error_ratio (how close predictions were to actual medals)
error_ratio = error_by_team / medals_by_team
# clean up NaN values
error_ratio[~pd.isnull(error_ratio)]
# clean up infinite values
error_ratio = error_ratio[np.isfinite(error_ratio)]
print("\nError Ratio:\n", error_ratio)

error_ratio.plot.hist()
plt.show()

# high error ratios for smaller countries; lower ratios for bigger countries
print("\nSort error ratios by country:\n", error_ratio.sort_values())
