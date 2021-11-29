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

def modelDropParams(model, x_train, y_train, x_test, y_test, dropCols=[]):
    """Function running model dropping some X-params."""
    
    print(f"\nResults without {dropCols}")
    x_train = x_train.copy().drop(columns=dropCols)
    model.fit(x_train, y_train)

    x_test = x_test.copy().drop(columns=dropCols)
    y_predict = model.predict(x_test)

    print(f'Accuracy LDA/QDA: \t\t {np.mean(y_predict == y_test):.2f}')
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

practiseTrain = pd.read_csv("train.csv")
practiceTest = pd.read_csv("test.csv")

# split practiseTrain into train & test
trainTestRatio = 0.95
trainIndex = np.random.choice(practiseTrain.shape[0], size=int(len(practiseTrain)*trainTestRatio), replace=False)
train = practiseTrain.iloc[trainIndex]
test = practiseTrain.iloc[~practiseTrain.index.isin(trainIndex)]

# split into X and y

x_train = train.copy().drop(columns=["Lead"])      # target
y_train = train["Lead"]
x_test = test.copy().drop(columns=["Lead"])
y_test = test["Lead"]

#x_train_LDA = practiseTrain.copy().drop(columns=["Lead"])      # target
#y_train_LDA = practiseTrain["Lead"]

model = skl_da.LinearDiscriminantAnalysis()
testParams = [["Year"], ["Gross"], ["Number words female", "Number words male"]]
combos = allCombos(testParams)
for c in combos:
    modelDropParams(model, x_train, y_train, x_test, y_test, dropCols=c)

'''
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
'''

'''
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
'''








