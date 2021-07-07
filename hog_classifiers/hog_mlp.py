import numpy as np
import joblib
import os

from sklearn.neural_network import MLPClassifier

from hog_load_data import hog_load_data
from hog_confusion_metrics import cal_confusion

(train_gradient_list, train_labels), (test_gradient_list, test_labels), (train_pca1, test_pca1), (
train_pca2, test_pca2) = hog_load_data()

hog_mlp_path = './hog_models/mlp/'
if (os.path.exists(hog_mlp_path) == False):
    os.mkdir(hog_mlp_path)

# Model path
hog_mlp_model = hog_mlp_path + 'hog_knn.model'
hog_mlp_pca_model1 = hog_mlp_path + 'hog_mlp_pca1.model'
hog_mlp_pca_model2 = hog_mlp_path + 'hog_mlp_pca2.model'


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
run_mlp(train_gradient_list, train_labels, test_gradient_list, test_labels, hog_mlp_model)
print()

print('--- Feature decomposed to 128 ---')
run_mlp(train_pca1, train_labels, test_pca1, test_labels, hog_mlp_pca_model1)
print()

print('--- Feature decomposed to 512 ---')
run_mlp(train_pca2, train_labels, test_pca2, test_labels, hog_mlp_pca_model2)
print()
