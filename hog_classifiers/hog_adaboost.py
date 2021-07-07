from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier as DTF
from sklearn.metrics import accuracy_score
import os
import numpy as np
import joblib

from hog_load_data import hog_load_data
from hog_confusion_metrics import cal_confusion

(train_gradient_list, train_labels), (test_gradient_list, test_labels), (train_pca1, test_pca1), (train_pca2, test_pca2) = hog_load_data()

hog_adaboost_path = './hog_models/adaboost/'
if (os.path.exists(hog_adaboost_path) == False):
    os.mkdir(hog_adaboost_path)

# Model path
hog_adaboost_model = hog_adaboost_path + 'hog_adaboost.model'
hog_adaboost_pca_model1 = hog_adaboost_path + 'hog_adaboost_pca1.model'
hog_adaboost_pca_model2 = hog_adaboost_path + 'hog_adaboost_pca2.model'

def run_adaboost(xtrain, ytrain, xtest, ytest, model, retry=True):
    # Adaboost train
    if (os.path.exists(model) == False or retry == True):
        print('---Adaboost train start---')
        Adaboost = AdaBoostClassifier(base_estimator=DTF(max_depth=4), n_estimators=200)
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
    adaboost_pred = adaboost_test.predict(xtest)
    cal_confusion(ytest, adaboost_pred)
    print('---Adaboost test end---')


print('--- Original Data ---')
run_adaboost(train_gradient_list, train_labels, test_gradient_list, test_labels, hog_adaboost_model)
print()

print('--- Feature decomposed to 128 ---')
run_adaboost(train_pca1, train_labels, test_pca1, test_labels, hog_adaboost_pca_model1)
print()

print('--- Feature decomposed to 512 ---')
run_adaboost(train_pca2, train_labels, test_pca2, test_labels, hog_adaboost_pca_model2)
print()