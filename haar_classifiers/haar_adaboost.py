from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier as DTF
from sklearn.metrics import accuracy_score
import os
import numpy as np
import joblib

from haar_load_data import haar_load_data
from haar_confusion_metrics import cal_confusion

(train_data_list, train_labels), (test_data_list, test_labels) = haar_load_data()

haar_adaboost_path = './haar_models/adaboost/'
if (os.path.exists(haar_adaboost_path) == False):
    os.mkdir(haar_adaboost_path)

# Model path
haar_adaboost_model = haar_adaboost_path + 'haar_adaboost.model'


def run_adaboost(xtrain, ytrain, xtest, ytest, model, retry=True):
    # Adaboost train
    if (os.path.exists(model) == False or retry == True):
        print('---Adaboost train start---')
        Adaboost = AdaBoostClassifier(base_estimator=DTF(max_depth=2), n_estimators=200)
        print(np.shape(np.asarray(xtrain)))
        Adaboost.fit(xtrain, ytrain)
        score = Adaboost.score(xtrain, ytrain)
        print('Adaboost train score', round(score, 3))
        joblib.dump(Adaboost, model)
        print('---Adaboost train end---')
    else:
        print('--- Adaboost train score loading ---')
        adaboost_train = joblib.load(model)
        score = adaboost_train.score(xtrain, ytrain)
        print('Adaboost train score', round(score, 3))
        print('--- Adaboost train score done ---')

    # Adaboost test
    adaboost_test = joblib.load(model)
    print('---Adaboost test start---')
    test_score = adaboost_test.score(xtest, ytest)
    print('Adaboost test score: ', round(test_score, 3))
    print('Adaboost Confusion Metrics and Reports')
    ada_pred = adaboost_test.predict(xtest)
    cal_confusion(ytest, ada_pred)
    print('---Adaboost test end---')


print('--- Original Data ---')
run_adaboost(train_data_list, train_labels, test_data_list, test_labels, haar_adaboost_model)
print()

