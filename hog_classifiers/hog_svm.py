import numpy as np
import joblib
import os
from sklearn import svm

from hog_load_data import hog_load_data
from hog_confusion_metrics import cal_confusion

(train_gradient_list, train_labels), (test_gradient_list, test_labels), (train_pca1, test_pca1), (train_pca2, test_pca2) = hog_load_data()

hog_svm_path = './hog_models/svm/'
if (os.path.exists(hog_svm_path) == False):
    os.mkdir(hog_svm_path)

# Model path
hog_svm_model = hog_svm_path + 'hog_svm.model'
hog_svm_pca_model1 = hog_svm_path + 'hog_svm_pca1.model'
hog_svm_pca_model2 = hog_svm_path + 'hog_svm_pca2.model'


def run_svm(xtrain, ytrain, xtest, ytest, model, retry=True):
    # SVM train
    if (os.path.exists(model) == False or retry==True):
        print('---SVM train start---')
        SVM = svm.NuSVC(kernel='linear')
        print(np.shape(np.asarray(xtrain)))
        SVM.fit(xtrain, ytrain)
        score = SVM.score(xtrain, ytrain)
        print('SVM train score', round(score, 3))
        joblib.dump(SVM, model)
        print('---SVM train end---')
    else:
        print('--- SVM train score loading ---')
        svm_train = joblib.load(model)
        score = svm_train.score(xtrain, ytrain)
        print('SVM train score', round(score, 3))
        print('--- SVM train score done ---')

    # SVM test
    svm_test = joblib.load(model)
    print('---SVM test start---')
    test_score = svm_test.score(xtest, ytest)
    print('SVM test score: ', round(test_score, 3))
    print('SVM Confusion Metrics and Reports')
    svm_pred = svm_test.predict(xtest)
    cal_confusion(ytest, svm_pred)
    print('---SVM test end---')


print('--- Original Data ---')
run_svm(train_gradient_list, train_labels, test_gradient_list, test_labels, hog_svm_model, retry=True)
print()

print('--- Feature decomposed to 128 ---')
run_svm(train_pca1, train_labels, test_pca1, test_labels, hog_svm_pca_model1, retry=True)
print()

print('--- Feature decomposed to 512 ---')
run_svm(train_pca2, train_labels, test_pca2, test_labels, hog_svm_pca_model2, retry=True)
print()
