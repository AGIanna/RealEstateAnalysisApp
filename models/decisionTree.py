import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold # abc
from sklearn import metrics

def decisionTree(df,columns): 
    models = []
    pred   = []
    score  = []
    conf   = []
    
    X = df[columns]
    Y = df["Label"]
    
    sfolder = StratifiedKFold(n_splits = 5,shuffle = False)
    for train_index, test_index, in sfolder.split(X, Y): 
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(X_train,Y_train.values)
        
        models.append(clf)
        pred.append(clf.predict(X_test))
        i = len(pred)
        conf.append(metrics.confusion_matrix(Y_test,pred[i-1]))
        score.append(metrics.accuracy_score(Y_test,pred[i-1]))
        
    return models, pred, conf, score

 

if __name__ == '__main__':
    
    filename = 'realestatedata_final.csv'

    traningdata = pd.read_csv('data/'+filename)
    traningdata.dropna(inplace=True)
    
    XClass_DT = ['ZipCode','ZHVI','Unemp Rate']
    
    model, pred, conf, score = decisionTree(traningdata, XClass_DT)
    
    print("Confusion Matrix of Decision Tree: ")
    print(conf)
    print()

    print("Accquarcy of Decision Tree: ")
    print(score)
    print()
    

