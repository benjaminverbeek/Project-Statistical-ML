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

import graphviz

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

if __name__=="__main__":
    practiseTrain = pd.read_csv("train.csv")

<<<<<<< HEAD
# split practiseTrain into train & test
trainTestRatio = 0.75
trainIndex = np.random.choice(practiseTrain.shape[0], size=int(len(practiseTrain)*trainTestRatio), replace=False)
train = practiseTrain.iloc[trainIndex]
test = practiseTrain.iloc[~practiseTrain.index.isin(trainIndex)]
=======
    # split practiseTrain into train & test
    trainTestRatio = 0.6
    trainIndex = np.random.choice(practiseTrain.shape[0], size=int(len(practiseTrain)*trainTestRatio), replace=False)
    train = practiseTrain.iloc[trainIndex]
    test = practiseTrain.iloc[~practiseTrain.index.isin(trainIndex)]
>>>>>>> 68af3a2da6e4b21f39875e354cba7182600a5cfd

    # split into X and y
    X_train = train.copy().drop(columns=["Lead"])      # target
    y_train = train["Lead"]
    X_test = test.copy().drop(columns=["Lead"])
    y_test = test["Lead"]

    # for final output predicition (has no y)
    finalTest = pd.read_csv("test.csv")
    X_finalTest = finalTest.copy()

<<<<<<< HEAD
#model = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=1)       # no better than random
model = RandomForestClassifier(max_depth=10, min_samples_leaf=1)           # Random forest: naive gives 80-85%
# TODO: How to improve?
testParams = [["Year"], ["Gross"], ["Number words female", "Number words male"]]
combos = allCombos(testParams)
print(f"Generated {len(combos)} combinations.")
print("Running ML-algo. for all combos.")

for c in combos:
    modelDropParams(model, X_train, y_train, X_test, y_test, dropCols=c)

specialTest = ["Gross"]
modelDropParams(model, X_train, y_train, X_test, y_test, dropCols=specialTest)

#while True:
#    exec(input("> "))
=======
    #model = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=1)       # no better than random
    model = RandomForestClassifier(max_depth=10, min_samples_leaf=1)           # Random forest: naive gives 80-85%
    # TODO: How to improve?
    testParams = [["Year"], ["Gross"], ["Number words female", "Number words male"]]
    combos = allCombos(testParams)
    print(f"Generated {len(combos)} combinations.")
    print("Running ML-algo. for all combos.")
    model = RandomForestClassifier(max_depth=10, min_samples_leaf=1)
    for c in combos:
        modelDropParams(model, X_train, y_train, X_test, y_test, dropCols=c)

    #while True:
    #    exec(input("> "))
>>>>>>> 68af3a2da6e4b21f39875e354cba7182600a5cfd
