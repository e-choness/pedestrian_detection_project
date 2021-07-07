import numpy as np
import joblib
import os

from sklearn.neighbors import KNeighborsClassifier
from haar_confusion_metrics import cal_confusion

from haar_load_data import haar_load_data

(train_data_list, train_labels), (test_data_list, test_labels) = haar_load_data()

haar_knn_path = './haar_models/knn/'
if (os.path.exists(haar_knn_path) == False):
    os.mkdir(haar_knn_path)

# Model path
haar_knn_model = haar_knn_path + 'haar_knn.model'


def run_knn(xtrain, ytrain, xtest, ytest, model, retry=True):
    # KNN train
    if (os.path.exists(model) == False or retry == True):
        print('---KNN train start---')
        KNN = KNeighborsClassifier(n_neighbors=3, n_jobs=1)
        print(np.shape(np.asarray(xtrain)))
        KNN.fit(xtrain, ytrain)
        score = KNN.score(xtrain, ytrain)
        print('KNN train score', round(score, 3))
        joblib.dump(KNN, model)
        print('---KNN train end---')
    else:
        print('--- KNN train score loading ---')
        knn_train = joblib.load(model)
        score = knn_train.score(xtrain, ytrain)
        print('KNN train score', round(score, 3))
        print('--- KNN train score done ---')

    # KNN test
    knn_test = joblib.load(model)
    print('---KNN test start---')
    test_score = knn_test.score(xtest, ytest)
    print('KNN test score: ', round(test_score, 3))
    print('KNN Confusion Metrics and Reports')
    knn_pred = knn_test.predict(xtest)
    cal_confusion(ytest, knn_pred)
    print('---KNN test end---')


print('--- Original Data ---')
run_knn(train_data_list, train_labels, test_data_list, test_labels, haar_knn_model)
print()


