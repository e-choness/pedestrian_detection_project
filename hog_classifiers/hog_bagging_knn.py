import numpy as np
import joblib
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from hog_load_data import hog_load_data
from hog_confusion_metrics import cal_confusion

(train_gradient_list, train_labels), (test_gradient_list, test_labels), (train_pca1, test_pca1), (train_pca2, test_pca2) = hog_load_data()

hog_bg_path = './hog_models/bagging/'
if (os.path.exists(hog_bg_path) == False):
    os.mkdir(hog_bg_path)

# Model path
hog_bg_knn_model = hog_bg_path + 'hog_bg_knn.model'
hog_bg_knn_pca_model1 = hog_bg_path + 'hog_bg_knn_pca1.model'
hog_bg_knn_pca_model2 = hog_bg_path + 'hog_bg_knn_pca2.model'


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
    bagging_knn_pred = bagging_knn_test.predict(xtest)
    cal_confusion(ytest, bagging_knn_pred)
    print('---Bagging-KNN test end---')


print('--- Original Data ---')
run_bagging_knn(train_gradient_list, train_labels, test_gradient_list, test_labels, hog_bg_knn_model)
print()

print('--- Feature decomposed to 128 ---')
run_bagging_knn(train_pca1, train_labels, test_pca1, test_labels, hog_bg_knn_pca_model1)
print()

print('--- Feature decomposed to 512 ---')
run_bagging_knn(train_pca2, train_labels, test_pca2, test_labels, hog_bg_knn_pca_model2)
print()
