# Classification LDA and QDA

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

import graphviz

practiseTrain = pd.read_csv("train.csv")

print('hello')
'''
# split into X and y
X_train = train.copy().drop(columns=["Lead"])      # target
y_train = train["Lead"]
X_test = test.copy().drop(columns=["Lead"])
y_test = test["Lead"]
'''