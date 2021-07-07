import numpy as np
import joblib
import os
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

from hog_load_data import hog_load_data
from hog_confusion_metrics import cal_confusion

(train_gradient_list, train_labels), (test_gradient_list, test_labels), (train_pca1, test_pca1), (
train_pca2, test_pca2) = hog_load_data()
print(np.unique(train_labels, return_counts=True))

print(train_gradient_list.shape)
print(train_labels.shape)
print(test_gradient_list.shape)
print(test_labels.shape)

hog_bayes_path = './hog_models/bayes/'
if (os.path.exists(hog_bayes_path) == False):
    os.mkdir(hog_bayes_path)

# Model path
hog_bayes_model = hog_bayes_path + 'hog_bayes.model'
hog_bayes_pca_model1 = hog_bayes_path + 'hog_bayes_pca1.model'
hog_bayes_pca_model2 = hog_bayes_path + 'hog_bayes_pca2.model'


def run_bayes(xtrain, ytrain, xtest, ytest, model, retry=True):
    # Bayes train
    if (os.path.exists(model) == False or retry==True):
        print('---Bayes train start---')
        Bayes = GaussianNB()
        print(np.shape(np.asarray(xtrain)))
        Bayes.fit(xtrain, ytrain)
        score = Bayes.score(xtrain, ytrain)
        print('Bayes train score', round(score, 3))
        joblib.dump(Bayes, model)
        print('---Bayes train end---')
    else:
        print('--- Bayes train score loading ---')
        bayes_train = joblib.load(model)
        score = bayes_train.score(xtrain, ytrain)
        print('Bayes train score', round(score, 3))
        print('--- Bayes train score done ---')

    # Bayes test
    bayes_test = joblib.load(model)
    print('---Bayes test start---')
    test_score = bayes_test.score(xtest, ytest)
    print('Bayes test score: ', round(test_score, 3))
    print('Bayes Confusion Metrics and Reports')
    bayes_pred = bayes_test.predict(xtest)
    cal_confusion(ytest, bayes_pred)
    print('---Bayes test end---')


print('--- Original Data ---')
run_bayes(train_gradient_list, train_labels, test_gradient_list, test_labels, hog_bayes_model)
print()

print('--- Feature decomposed to 128 ---')
run_bayes(train_pca1, train_labels, test_pca1, test_labels, hog_bayes_pca_model1)
print()

print('--- Feature decomposed to 512 ---')
run_bayes(train_pca2, train_labels, test_pca2, test_labels, hog_bayes_pca_model2)
print()
