import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
from sklearn.preprocessing import StandardScaler

def modelDropParams(model, X_train_scaled_df, y_train, X_test_scaled_df, y_test, dropCols=[]):
    """Function running model dropping some X-params."""
    
    print(f"\n\nResults without {dropCols}")
    X_train_scaled_df = X_train_scaled_df.copy().drop(columns=dropCols)
    model.fit(X_train_scaled_df, y_train)

    X_test_scaled_df = X_test_scaled_df.copy().drop(columns=dropCols)
    y_predict = model.predict(X_test_scaled_df)

    print(f'Regression accuracy: \t {np.mean(y_predict == y_test):.2f}')
    allMale = y_test.copy().replace(["Female"],"Male") # make a copy with all Male.
    print(f'Accuracy all-male: \t {np.mean(allMale == y_test):.2f}')
    print(pd.crosstab(y_predict, y_test))
    print("-------------------------------------------------------------------------------")
    features = X_train_scaled_df.columns.values
    Summary_of_Table = pd.DataFrame(columns=['Feature Name'], data=features)
    Summary_of_Table['Coefficient'] = np.transpose(model.coef_)
    print(Summary_of_Table)

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

practiseTrain = pd.read_csv("~/SML/Project-Statistical-ML/train.csv")

# split practiseTrain into train & test
trainTestRatio = 0.6
trainIndex = np.random.choice(practiseTrain.shape[0], size=int(len(practiseTrain)*trainTestRatio), replace=False)
train = practiseTrain.iloc[trainIndex]
test = practiseTrain.iloc[~practiseTrain.index.isin(trainIndex)]

# split into X and y and apply scaling where the sum of a column is zero
X_train = train.copy().drop(columns=["Lead"])      # target
X_train_scaled_df = rescaleDataFrame(X_train) 

X_test = test.copy().drop(columns=["Lead"])
X_test_scaled_df = rescaleDataFrame(X_test) 

y_train = train["Lead"]
y_test = test["Lead"]

# Confirm that df is transformed
# print(X_train_scaled_df.head())
# print(X_test_scaled_df.head())

# for final output predicition (has no y)
finalTest = pd.read_csv("~/SML/Project-Statistical-ML/test.csv")
X_finalTest = finalTest.copy()

X_finalTest_scaled = StandardScaler().fit_transform(X_finalTest.values)
X_finalTest_scaled_df = pd.DataFrame(X_finalTest_scaled, index=X_finalTest.index, columns=X_finalTest.columns)

model = skl_lm.LogisticRegression(solver='lbfgs', C=12, random_state=0) 
# TODO: How to improve?
testParams = [["Year"], ["Gross"], ["Number words female", "Number words male"]]
combos = allCombos(testParams)
print(f"Generated {len(combos)} combinations.")
print("Running ML-algo. for all combos.")
for c in combos:
    modelDropParams(model, X_train_scaled_df, y_train, X_test_scaled_df, y_test, dropCols=c)

