import pandas as pd
import numpy as np
import sklearn.linear_model as skl_lm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
import sklearn.model_selection as skl_ms
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import sklearn.discriminant_analysis as skl_da

def modelDropParams(model, X_train, y_train, X_test, y_test, dropCols=[]):
    """Function running model dropping some X-params."""
    
    print(f"\n\nResults without {dropCols}")
    X_train= X_train.copy().drop(columns=dropCols)
    model.fit(X_train, y_train)

    X_test = X_test.copy().drop(columns=dropCols)
    y_predict = model.predict(X_test)
    print(pd.crosstab(y_predict, y_test))
    #print(f'Regression accuracy: \t {np.mean(y_predict == y_test):.2f}')
    #allMale = y_test.copy().replace(["Female"],"Male") # make a copy with all Male.
    #print(f'Accuracy all-male: \t {np.mean(allMale == y_test):.2f}')
    
    #print("-------------------------------------------------------------------------------")
    #features = X_train_scaled_df.columns.values
    #Summary_of_Table = pd.DataFrame(columns=['Feature Name'], data=features)
    #Summary_of_Table['Coefficient'] = np.transpose(model.coef_)
    #print(Summary_of_Table)

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

def labelmaker(testParams, modelNames, yes = False):
    if yes:
        label = ["All Male"]
        for i in range(len(allCombos(testParams))):
            label.extend(modelNames)
        return label
    else:
        return None

def crossValidationDropper(models = [], train0 = pd.read_csv('train.csv'),  cDrop = [], rState = None, n_fold = 10):
    """Takes an array of models and a training dataset"""
    X = rescaleDataFrame(train0.drop(columns = "Lead"))
    y = train0["Lead"]

    combos = allCombos(cDrop)
    print(combos)

    
    cv = skl_ms.KFold(n_splits=n_fold, random_state= rState, shuffle=True)

    accuracy = np.zeros((len(models)*len(combos) + 1, n_fold)).T
    f_accuracy = np.zeros((len(models)*len(combos) + 1, n_fold)).T

    for i, index in enumerate(cv.split(X)):
        X_train = X.iloc[index[0]]
        y_train = y.iloc[index[0]]
        X_test = X.iloc[index[1]]
        y_test = y.iloc[index[1]]
        
        j = 1
        accuracy[i, 0]  = np.mean("Male" == y_test) 
        #f_accuracy[i, 0] = np.mean("Female" == "Male")

        for c in combos:
            X_testC = X_test.copy().drop(columns=c)
            X_trainC= X_train.copy().drop(columns=c)
            for m in models:               
                m.fit(X_trainC, y_train)
                y_predict = m.predict(X_testC)
                
                d = 0
                n = 0
                for a, b in zip(y_predict, y_test):
                    if b == "Female":
                        d += 1
                        if a == b:
                            n += 1

                accuracy[i,j] = np.mean(y_predict == y_test)
                f_accuracy[i,j] = n/d

                #modelDropParams(m,X_train,y_train, X_test, y_test, c)
                j += 1
                 
    return accuracy, f_accuracy

if __name__ == "__main__":
    train0 = pd.read_csv('train.csv')

    models = []
    """
    for i in range(10):
        
    """
    #for i in range(10):
    
    M0 = "Q2" # or "G" or "A"
    n0 = 500
    nk = 0
    l0 = 0.4
    lk = 0
    m0 = 0.5
    mk = 0

    for i in range(1):
        #models.append(AdaBoostClassifier(n_estimators = 100, learning_rate = 1 + i*0.1 ,random_state = None))
        #models.append(AdaBoostClassifier(n_estimators = 10*(i+1), learning_rate = 1 ,random_state = 1))
        #models.append(GradientBoostingClassifier(n_estimators = 20*(i+11), learning_rate = 1.2 ))
        #models.append(GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.1*(i+7)))
        #models.append(GradientBoostingClassifier(n_estimators = n0 + nk*i, learning_rate = l0 + lk*i, min_samples_split = m0 + mk*i))
        pass

    #models.append(skl_lm.LogisticRegression(solver='lbfgs', C=12, random_state = 1))
    #models.append(skl_da.LinearDiscriminantAnalysis())
    models.append(skl_da.QuadraticDiscriminantAnalysis())
    #models.append(RandomForestClassifier(max_depth=10, min_samples_leaf=1))

    testParams = [["Year"], ["Gross"], ["Number words female", "Number words male"]]
    #testParams = []

    r, fr = crossValidationDropper(models, train0, testParams)

    moreTimes = 10
    for i in range(moreTimes):
        a, fa = crossValidationDropper(models, train0, testParams)
        r = np.append(r, a, axis = 0)
        fr = np.append(fr, fa, axis = 0)

    #Plot:
    modelNames = ["LR", "Boosting", "LDA", "Random Forest"]
    f1 = plt.figure()
    plt.boxplot(r)
    ax = plt.gca()
    ax.set(xlabel="1")

    f2 = plt.figure()
    plt.boxplot(fr, labels = labels)
    fileEnd = f'-{M0}-{n0}+{nk}-{l0}+{lk}-x{moreTimes}-{m0}+{mk}'
    fileEnd = f'{M0}'
    f1.savefig("SML/Graphs/ac"+fileEnd+".png")
    f2.savefig("SML/Graphs/fac"+fileEnd+".png")
    plt.show()

    
