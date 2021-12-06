import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
# For balancing trainingdata. "Synthetic Minority Over-sampling TEchnique"
from imblearn.over_sampling import SMOTE

def crossVal(model, X, y, print_accuracy=True, run_all_models=False, dropCols=[]):
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

    # Iterate over all k-folds, fit model and sum to cumulative confusion matrix
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

    print(tot_crosstab)
    # Statistics:
    acc = (tot_crosstab['Female'][0] + tot_crosstab['Male'][1]) / tot_crosstab.values.sum()
    nFem = tot_crosstab['Female'].values.sum()
    nMale = tot_crosstab['Male'].values.sum()
    accFem = tot_crosstab['Female'][0] / nFem
    accMale = tot_crosstab['Male'][1] / nMale
    percentMale = nMale / (nMale + nFem)

    if print_accuracy:    
        print(f'Accuracy: {acc:.5f}')
        print(f'Accuracy Female / Male:\t {accFem:.5f} / {accMale:.5f} \t (testdata contains {percentMale:.5f} % males)')
    
    if run_all_models:
        results.append((acc, accFem, accMale, model, dropCols))


def modelDropParams(model, X, y, dropCols=[], run_all_models=False):
    """Function running model dropping some X-params. With cross-validation."""
    
    print(f"\nResults without {dropCols}")
    X = X.copy().drop(columns=dropCols)

    crossVal(model, X, y, run_all_models=run_all_models, dropCols=dropCols)

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

def rescaleDataFrame(df):
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_input, index=df.index, columns=df.columns)
    return scaled_df
######

# Read the files into data frames 
practiseTrain = pd.read_csv("train.csv")
practiceTest = pd.read_csv("test.csv")

# Split data into two frames, X and y
X = practiseTrain.copy().drop(columns=["Lead"])      # target
y = practiseTrain["Lead"]

# Rescale dataframe, can be commented to test if it gives better results or not
X = rescaleDataFrame(X) 

# Dict with models
models = {
        'LDA': skl_da.LinearDiscriminantAnalysis(),
        'QDA': skl_da.QuadraticDiscriminantAnalysis(),
        'tree': tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=1),
        'random-forest': RandomForestClassifier(max_depth=5, min_samples_leaf=1),
        'logreg': skl_lm.LogisticRegression(solver='lbfgs', C=12, random_state=0)
    }

model = models['random-forest']
run_all_models = True

# Declare parameters to evaluate and extract all combos
testParams = [["Year"], ["Gross"], ["Number words female", "Number words male"]]
combos = allCombos(testParams)
print(f"Generated {len(combos)} combinations.")
print("Running ML-algo. for all combos.")

# TODO: possibly add output to excel for easier report-writing? Or all just take their model and write.
# OR save results to a dict and find max accuracy.
# Iterate over all combos 


if run_all_models:
    results = []
    for model in models.values():
        print(f'----- RUNNING MODEL: {model} -----')
        for c in combos:
            modelDropParams(model, X, y, dropCols=c, run_all_models=True)
    print('\n#### FINAL RESULTS ####')
    print(f'Top 5 by total accuracy: \n')
    print(*list(reversed(sorted(results)[-5:])), sep='\n')  # prints line-by-line
    print('###########')
    print(f'Worst 5 by total accuracy: \n')
    print(*sorted(results)[:5], sep='\n')
else:
    for c in combos:
        modelDropParams(model, X, y, dropCols=c)
