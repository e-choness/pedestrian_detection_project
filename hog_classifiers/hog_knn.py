import numpy as np
import joblib
import os

from sklearn.neighbors import KNeighborsClassifier

from hog_load_data import hog_load_data
from hog_confusion_metrics import cal_confusion

(train_gradient_list, train_labels), (test_gradient_list, test_labels), (train_pca1, test_pca1), (train_pca2, test_pca2) = hog_load_data()

hog_knn_path = './hog_models/knn/'
if (os.path.exists(hog_knn_path) == False):
    os.mkdir(hog_knn_path)

# Model path
hog_knn_model = hog_knn_path + 'hog_knn.model'
hog_knn_pca_model1 = hog_knn_path + 'hog_knn_pca1.model'
hog_knn_pca_model2 = hog_knn_path + 'hog_knn_pca2.model'

def run_knn(xtrain, ytrain, xtest, ytest, model, retry=True):
    # KNN train
    if (os.path.exists(model) == False or retry == True):
        print('---KNN train start---')
        KNN = KNeighborsClassifier(n_neighbors=3, n_jobs=1)
        print(np.shape(np.asarray(xtrain)))
        KNN.fit(xtrain, ytrain)
        score = KNN.score(xtrain, ytrain)
        print('KNN train score', round(score,3))
        joblib.dump(KNN, model)
        print('---KNN train end---')
    else:
        print('--- KNN train score loading ---')
        knn_train = joblib.load(model)
        score = knn_train.score(xtrain, ytrain)
        print('KNN train score', round(score,3))
        print('--- KNN train score done ---')

    # KNN test
    knn_test = joblib.load(model)
    print('---KNN test start---')
    test_score = knn_test.score(xtest, ytest)
    print('KNN test score: ', round(test_score,3))
    print('KNN Confusion Metrics and Reports')
    knn_pred = knn_test.predict(xtest)
    cal_confusion(ytest, knn_pred)
    print('---KNN test end---')


print('--- Original Data ---')
run_knn(train_gradient_list, train_labels, test_gradient_list, test_labels, hog_knn_model)
print()

print('--- Feature decomposed to 128 ---')
run_knn(train_pca1, train_labels, test_pca1, test_labels, hog_knn_pca_model1)
print()

print('--- Feature decomposed to 512 ---')
run_knn(train_pca2, train_labels, test_pca2, test_labels, hog_knn_pca_model2)
print()
