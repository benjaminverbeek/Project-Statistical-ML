# Tree method for movie leads
# Project in Statistical Machine Learning, 5 c
# Written by Benjamin Verbeek, Uppsala 2021-11-24

# NOTES: Just a decision tree performs no better than guessing "Male" on all.
# Using randomForest, accuracy is roughly 85% on data with 78% male (so 
# slighyly better only). Try with bagging? Some other methods? Tuning tree?
# Data is very unbalanced; much more male data.

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE

import graphviz

# Normalize input data to be between 0 and 1: imbalance in order of
# magnitudes messes up weight of different factors. Therefore,
# removing some inputs improves output.
# need to use same factor for normalization during train and test 
# (and later tests), even though norm wont be perfect.
# Try it!

# Average of one type, divide by that.
# 

def modelDropParams(model, X_train, y_train, X_test, y_test, dropCols=[]):
    """Function running model dropping some X-params."""
    
    print(f"\nResults without {dropCols}")
    X_train = X_train.copy().drop(columns=dropCols)
    model.fit(X_train, y_train)

    X_test = X_test.copy().drop(columns=dropCols)
    y_predict = model.predict(X_test)

    print(f'Accuracy tree: \t\t {np.mean(y_predict == y_test):.2f}')
    allMale = y_test.copy().replace(["Female"],"Male") # make a copy with all Male.
    print(f'Accuracy all-male: \t {np.mean(allMale == y_test):.2f}')
    print(pd.crosstab(y_predict, y_test))
    print("-----------------")


def allCombos(lst):
    """Takes in a list of lists and returns a list of all combinations of list elements."""
    combos = []
    for i in range(2**len(lst)):
        a=i
        params = []
        for j in range(len(lst)):
            if a%2 == 1:
                params += lst[j]
            a = a//2
        combos.append(params)
    return combos
######

#if __name__=="__main__":
practiseTrain = pd.read_csv("train.csv")
practiseTrainNorm = practiseTrain.copy()

# Normalized... does not seem to affect.
for c in practiseTrainNorm.copy().drop(columns=["Lead"]).columns:
    m = practiseTrainNorm[c].mean()
    practiseTrainNorm[c] = practiseTrainNorm[c].div(m)
    m = practiseTrainNorm[c].mean()
    practiseTrainNorm[c] = practiseTrainNorm[c].subtract(m)

# split practiseTrain into train & test
trainTestRatio = 0.6
trainIndex = np.random.choice(practiseTrain.shape[0], size=int(len(practiseTrain)*trainTestRatio), replace=False)
train = practiseTrain.iloc[trainIndex]
test = practiseTrain.iloc[~practiseTrain.index.isin(trainIndex)]
# Normed
trainNorm = practiseTrainNorm.iloc[trainIndex]
testNorm = practiseTrainNorm.iloc[~practiseTrainNorm.index.isin(trainIndex)]

# split into X and y
X_train = train.copy().drop(columns=["Lead"])      # target
y_train = train["Lead"]
X_test = test.copy().drop(columns=["Lead"])
y_test = test["Lead"]

### Resample training data to balance M/F
# Vastly improves performance on female classification (slightly dropping male)
sm = SMOTE(random_state=42)
X_resTrain, y_resTrain = sm.fit_resample(X_train, y_train)

X_train = X_resTrain
y_train = y_resTrain
####

X_trainNorm = trainNorm.copy().drop(columns=["Lead"])      # target
y_trainNorm = trainNorm["Lead"]
X_testNorm = testNorm.copy().drop(columns=["Lead"])
y_testNorm = testNorm["Lead"]

# for final output predicition (has no y)
finalTest = pd.read_csv("test.csv")
# NEED TO NORMALIZE
X_finalTest = finalTest.copy()

#model = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=1)       # no better than random
model = RandomForestClassifier(max_depth=10, min_samples_leaf=1)           # Random forest: naive gives 80-85%
# TODO: How to improve?
testParams = [["Year"], ["Gross"], ["Number words female", "Number words male"]]
combos = allCombos(testParams)
print(f"Generated {len(combos)} combinations.")
print("Running ML-algo. for all combos.")

for c in combos:
    model = RandomForestClassifier(max_depth=10, min_samples_leaf=1)
    model2 = RandomForestClassifier(max_depth=10, min_samples_leaf=1)
    modelDropParams(model, X_train, y_train, X_test, y_test, dropCols=c)
    modelDropParams(model2, X_trainNorm, y_trainNorm, X_testNorm, y_testNorm, dropCols=c)

#specialTest = ["Gross"]
#modelDropParams(model, X_train, y_train, X_test, y_test, dropCols=specialTest)

#while True:
#    exec(input("> "))
