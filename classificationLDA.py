# Classification LDA and QDA

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.model_selection import KFold

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

# Read the files into data frames 
practiseTrain = pd.read_csv("train.csv")
practiceTest = pd.read_csv("test.csv")

# Split data into two frames, X and y
X = practiseTrain.copy().drop(columns=["Lead"])      # target
y = practiseTrain["Lead"]

# Split index for the folds
kf = KFold(n_splits = 10, shuffle = True, random_state = 1)
testIndicies = [] 

# Choose model
#model = skl_da.LinearDiscriminantAnalysis()
model = skl_da.QuadraticDiscriminantAnalysis()

print(f"model is {model}")

all_predictions = []
all_ys = []
#all_predictions_pd = pd.crosstab(all_predictions, all_ys)

# initialize data of lists.
data = {'Female':[0, 0],
        'Male':[0, 0]}
  
# Create DataFrame
test_pd = pd.DataFrame(data, index=['Female', 'Male'])
data = {'Female':[0, 0],
            'Male':[0, 0]}
tot_crosstab = pd.DataFrame(data, index=['Female', 'Male'])

for train_index, test_index in kf.split(X):
    testIndicies.append(test_index)
    X_train, X_test = X.iloc[train_index,: ], X.iloc[test_index,: ]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)

    predict_prob = model.predict_proba(X_test)

    prediction = np.empty(len(X_test), dtype=object)
    prediction = np.where(predict_prob[:,0]>0.5, 'Female', 'Male')

    conf_mat = pd.crosstab(prediction, y_test)

    test_pd = test_pd + conf_mat

    print(conf_mat)

    '''#print(f"accuracy is {acc}\n")
    acc = (tot_crosstab['Female'][0] + tot_crosstab['Male'][1]) / tot_crosstab.values.sum()
    nFem = tot_crosstab['Female'].values.sum()
    nMale = tot_crosstab['Male'].values.sum()
    accFem = tot_crosstab['Female'][0] / nFem
    accMale = tot_crosstab['Male'][1] / nMale
    percentMale = nMale / (nMale + nFem)

      
    print(f'Accuracy: {acc:.5f}')
    print(f'Accuracy Female / Male:\t {accFem:.5f} / {accMale:.5f} \t (testdata contains {percentMale:.5f} % males)')'''

print(test_pd)

'''result = next(kf.split(X), None)

# Split both X and y into fold
X_train = X.iloc[result[0]]
X_test =  X.iloc[result[1]]
y_train = y.iloc[result[0]]
y_test =  y.iloc[result[1]]


print(f"model is {model}")
model.fit(X_train, y_train)

predict_prob = model.predict_proba(X_test)

prediction = np.empty(len(X_test), dtype=object)
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

# split practiseTrain into train & test
trainTestRatio = 0.6
trainIndex = np.random.choice(practiseTrain.shape[0], size=int(len(practiseTrain)*trainTestRatio), replace=False)
train = practiseTrain.iloc[trainIndex]
test = practiseTrain.iloc[~practiseTrain.index.isin(trainIndex)]

# split into X and y
x_train = train.copy().drop(columns=["Lead"])      # target
y_train = train["Lead"]
x_test = test.copy().drop(columns=["Lead"])
y_test = test["Lead"]

#model = skl_da.LinearDiscriminantAnalysis()
model = skl_da.QuadraticDiscriminantAnalysis()

testParams = [["Year"], ["Gross"], ["Number words female", "Number words male"]]
combos = allCombos(testParams)
for c in combos:
    modelDropParams(model, x_train, y_train, x_test, y_test, dropCols=c)
'''








