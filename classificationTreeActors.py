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

practiseTrain = pd.read_csv("train.csv")
finalTest = pd.read_csv("test.csv")

# split practiseTrain into train & test
trainTestRatio = 0.6
trainIndex = np.random.choice(practiseTrain.shape[0], size=int(len(practiseTrain)*trainTestRatio), replace=False)
train = practiseTrain.iloc[trainIndex]
test = practiseTrain.iloc[~practiseTrain.index.isin(trainIndex)]

X_train = train.copy().drop(columns=["Lead"])      # target
y_train = train["Lead"]

model = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=1)
model.fit(X=X_train, y=y_train)

X_finalTest = finalTest.copy()

X_test = test.copy().drop(columns=["Lead"])
y_test = test["Lead"]

y_predict = model.predict(X_test)
print(f'Accuracy: {np.mean(y_predict == y_test):.2f}')
pd.crosstab(y_predict, y_test)





#while True:
#    exec(input("> "))