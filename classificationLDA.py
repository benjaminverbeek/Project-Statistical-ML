# Classification LDA and QDA

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

import graphviz

'''
def predict(model, xTest, yTest):
      print(f"model is {model}")


  model.fit(xTrain, yTrain)

  pred_class = model.predict(xTest)
  predict_prob = model.predict_proba(xTest)

  prediction = np.empty(len(xTest), dtype=object)
  prediction = np.where(predict_prob[:,0]>0.5, 'benign', 'malignant')

  acc = np.mean(prediction==yTest)
  print(f"accuracy is {acc}")


  #confmat = pd.crosstab(pred_class, yTest)
  print(pd.crosstab(prediction, yTest))
'''

practiseTrain = pd.read_csv("train.csv")
practiceTest = pd.read_csv("test.csv")

# split practiseTrain into train & test
trainTestRatio = 0.9
trainIndex = np.random.choice(practiseTrain.shape[0], size=int(len(practiseTrain)*trainTestRatio), replace=False)
train = practiseTrain.iloc[trainIndex]
test = practiseTrain.iloc[~practiseTrain.index.isin(trainIndex)]

# split into X and y

# xTrain = train[['V3','V4','V5']]
x_train = train.copy().drop(columns=["Lead"])      # target
y_train = train["Lead"]
x_test = test.copy().drop(columns=["Lead"])
y_test = test["Lead"]

#x_train_LDA = practiseTrain.copy().drop(columns=["Lead"])      # target
#y_train_LDA = practiseTrain["Lead"]

#LDA
model = skl_da.LinearDiscriminantAnalysis()
print(f"model is {model}")
model.fit(x_train, y_train)

predict_prob = model.predict_proba(x_test)

prediction = np.empty(len(x_test), dtype=object)
prediction = np.where(predict_prob[:,0]>0.5, 'Female', 'Male')

acc = np.mean(prediction==y_test)
print(f"accuracy is {acc}\n")

print(pd.crosstab(prediction, y_test))

#QDA
model = skl_da.QuadraticDiscriminantAnalysis()
print(f"model is {model}")
model.fit(x_train, y_train)

predict_prob = model.predict_proba(x_test)

prediction = np.empty(len(x_test), dtype=object)
prediction = np.where(predict_prob[:,0]>0.5, 'Female', 'Male')

acc = np.mean(prediction==y_test)
print(f"accuracy is {acc}\n")

print(pd.crosstab(prediction, y_test))








