import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.model_selection import KFold

def crossVal(model, X, y):
    print(f"model is {model}")

    # Split index for the folds
    kf = KFold(n_splits = 10, shuffle = True, random_state = 1)
    testIndicies = [] 

    # Initiate cumulative sum variables
    all_predictions = []
    all_ys = []
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

        tot_crosstab = tot_crosstab + conf_mat

    #print(f"accuracy is {acc}\n")

    print(tot_crosstab)

def modelDropParams(model, X, y, dropCols=[]):
    """Function running model dropping some X-params."""
    
    print(f"\nResults without {dropCols}")
    X = X.copy().drop(columns=dropCols)

    crossVal(model, X, y)

    #print(f'Accuracy tree: \t\t {np.mean(y_predict == y_test):.2f}')
    #allMale = y_test.copy().replace(["Female"],"Male") # make a copy with all Male.
    
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




# Read the files into data frames 
practiseTrain = pd.read_csv("train.csv")
practiceTest = pd.read_csv("test.csv")

# Split data into two frames, X and y
X = practiseTrain.copy().drop(columns=["Lead"])      # target
y = practiseTrain["Lead"]

# Choose model
model = skl_da.LinearDiscriminantAnalysis()
#model = skl_da.QuadraticDiscriminantAnalysis()
#model = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=1)       # no better than random
#model = RandomForestClassifier(max_depth=10, min_samples_leaf=1)  

testParams = [["Year"], ["Gross"], ["Number words female", "Number words male"]]
combos = allCombos(testParams)
print(f"Generated {len(combos)} combinations.")
print("Running ML-algo. for all combos.")
for c in combos:
    modelDropParams(model, X, y, dropCols=c)