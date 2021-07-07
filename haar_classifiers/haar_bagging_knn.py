import numpy as np
import joblib
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from haar_load_data import haar_load_data
from haar_confusion_metrics import cal_confusion

(train_data_list, train_labels), (test_data_list, test_labels) = haar_load_data()

haar_bg_path = './haar_models/bagging/'
if (os.path.exists(haar_bg_path) == False):
    os.mkdir(haar_bg_path)

# Model path
haar_bg_knn_model = haar_bg_path + 'haar_bg_knn.model'


def run_bagging_knn(xtrain, ytrain, xtest, ytest, model, retry=True):
    # Bagging-KNN train
    if (os.path.exists(model) == False or retry == True):
        print('--- Bagging-KNN train start ---')
        bagging_knn = BaggingClassifier(KNeighborsClassifier(), n_estimators=10, max_samples=0.5, max_features=0.5)
        print(np.shape(np.asarray(xtrain)))
        bagging_knn.fit(xtrain, ytrain)
        score = bagging_knn.score(xtrain, ytrain)
        print('Bagging-KNN train score', round(score, 3))
        joblib.dump(bagging_knn, model)
        print('--- Bagging-KNN train end ---')
    else:
        print('--- Bagging-KNN train score loading ---')
        bagging_knn_train = joblib.load(model)
        score = bagging_knn_train.score(xtrain, ytrain)
        print('Bagging-KNN train score', round(score, 3))
        print('--- Bagging-KNN train score done ---')

    # Bagging-KNN test
    bagging_knn_test = joblib.load(model)
    print('---Bagging-KNN test start---')
    test_score = bagging_knn_test.score(xtest, ytest)
    print('Bagging-KNN test score: ', round(test_score, 3))
    print('Bagging-KNN Confusion Metrics and Reports')
    bg_pred = bagging_knn_test.predict(xtest)
    cal_confusion(ytest, bg_pred)
    print('---Bagging-KNN test end---')


print('--- Original Data ---')
run_bagging_knn(train_data_list, train_labels, test_data_list, test_labels, haar_bg_knn_model)
print()


