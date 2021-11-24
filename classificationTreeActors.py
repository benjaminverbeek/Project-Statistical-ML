# Decision tree for movie leads
# Project in Statistical Machine Learning, 5 c
# Written by Benjamin Verbeek, Uppsala 2021-11-24

# NOTES: Initial, naive implementation of calssification tree
# yields accuracy of around 75-80 % (depth=3). 

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

import graphviz

def treeDropCols(model, X_train, y_train, X_test, y_test, dropCols=[]):
    """Function dropping some X-data."""
    
    print(f"\nResults without {dropCols}")
    X_train = X_train.copy().drop(columns=dropCols)
    model.fit(X_train, y_train)

    X_test = X_test.copy().drop(columns=dropCols)
    y_predict = model.predict(X_test)

    print(f'Accuracy tree: \t\t {np.mean(y_predict == y_test):.2f}')
    print(f'Accuracy all-male: \t {np.mean(allMale == y_test):.2f}')
    print(pd.crosstab(y_predict, y_test))
    print("-----------------")

######

practiseTrain = pd.read_csv("train.csv")
finalTest = pd.read_csv("test.csv")

# split practiseTrain into train & test
trainTestRatio = 0.6
trainIndex = np.random.choice(practiseTrain.shape[0], size=int(len(practiseTrain)*trainTestRatio), replace=False)
train = practiseTrain.iloc[trainIndex]
test = practiseTrain.iloc[~practiseTrain.index.isin(trainIndex)]

X_train = train.copy().drop(columns=["Lead"])      # target
y_train = train["Lead"]

#model = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=1)
#model.fit(X=X_train, y=y_train)

X_finalTest = finalTest.copy()

X_test = test.copy().drop(columns=["Lead"])
y_test = test["Lead"]
allMale = test["Lead"].copy().replace(["Female"],"Male") # make a copy with all Male.

#y_predict = model.predict(X_test)
#print(f'Accuracy tree: \t\t {np.mean(y_predict == y_test):.2f}')
#print(f'Accuracy all-male: \t {np.mean(allMale == y_test):.2f}')
#print(pd.crosstab(y_predict, y_test))

model = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=1)
treeDropCols(model, X_train, y_train, X_test, y_test, dropCols=['Year'])
treeDropCols(model, X_train, y_train, X_test, y_test, dropCols=['Year'])

#while True:
#    exec(input("> "))