import numpy as np
import joblib
import os
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

# os.chdir('./')

from haar_load_data import haar_load_data
from haar_confusion_metrics import cal_confusion

(train_data_list, train_labels), (test_data_list, test_labels) = haar_load_data()

haar_bayes_path = './haar_models/bayes/'
if (os.path.exists(haar_bayes_path) == False):
    os.mkdir(haar_bayes_path)

# Model path
haar_bayes_model = haar_bayes_path + 'haar_bayes.model'


def run_bayes(xtrain, ytrain, xtest, ytest, model, retry=True):
    # Bayes train
    if (os.path.exists(model) == False or retry == True):
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
run_bayes(train_data_list, train_labels, test_data_list, test_labels, haar_bayes_model)
print()


