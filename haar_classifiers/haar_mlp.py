import numpy as np
import joblib
import os

from sklearn.neural_network import MLPClassifier
from haar_confusion_metrics import cal_confusion

from haar_load_data import haar_load_data

(train_data_list, train_labels), (test_data_list, test_labels) = haar_load_data()

haar_mlp_path = './haar_models/mlp/'
if (os.path.exists(haar_mlp_path) == False):
    os.mkdir(haar_mlp_path)

# Model path
haar_mlp_model = haar_mlp_path + 'haar_knn.model'

def run_mlp(xtrain, ytrain, xtest, ytest, model, retry=True):
    # MLP train
    if (os.path.exists(model) == False or retry == True):
        print('---MLP train start---')
        MLP = MLPClassifier(activation='tanh', solver='adam', alpha=0.0001, learning_rate='adaptive',
                            learning_rate_init=0.001, max_iter=200)
        print(np.shape(np.asarray(xtrain)))
        MLP.fit(xtrain, ytrain)
        score = MLP.score(xtrain, ytrain)
        print('MLP train score', round(score, 3))
        joblib.dump(MLP, model)
        print('---MLP train end---')
    else:
        print('--- MLP train score loading ---')
        mlp_train = joblib.load(model)
        score = mlp_train.score(xtrain, ytrain)
        print('MLP train score', round(score, 3))
        print('--- MLP train score done ---')

    # MLP test
    mlp_test = joblib.load(model)
    print('---MLP test start---')
    test_score = mlp_test.score(xtest, ytest)
    print('MLP test score: ', round(test_score, 3))
    print('MLP Confusion Metrics and Reports')
    mlp_pred = mlp_test.predict(xtest)
    cal_confusion(ytest, mlp_pred)
    print('---MLP test end---')


print('--- Original Data ---')
run_mlp(train_data_list, train_labels, test_data_list, test_labels, haar_mlp_model)
print()


