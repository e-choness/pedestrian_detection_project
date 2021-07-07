import numpy as np
import joblib
import os
from sklearn import svm

from haar_load_data import haar_load_data
from haar_confusion_metrics import cal_confusion

(train_data_list, train_labels), (test_data_list, test_labels) = haar_load_data()

# print(np.unique(test_labels))

haar_svm_path = './haar_models/svm/'
if (os.path.exists(haar_svm_path) == False):
    os.mkdir(haar_svm_path)

# Model path
haar_svm_model = haar_svm_path + 'haar_svm.model'


def run_svm(xtrain, ytrain, xtest, ytest, model, retry=True):
    # SVM train
    if (os.path.exists(model) == False or retry == True):
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
run_svm(train_data_list, train_labels, test_data_list, test_labels, haar_svm_model)
print()

