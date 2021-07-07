import numpy as np
import joblib
import os

from sklearn.tree import DecisionTreeClassifier

from hog_load_data import hog_load_data
from hog_confusion_metrics import cal_confusion


(train_gradient_list, train_labels), (test_gradient_list, test_labels), (train_pca1, test_pca1), (train_pca2, test_pca2) = hog_load_data()

hog_dt_path = './hog_models/decision_tree/'
if (os.path.exists(hog_dt_path) == False):
    os.mkdir(hog_dt_path)

# Model path
hog_dt_model = hog_dt_path + 'hog_dt.model'
hog_dt_pca_model1 = hog_dt_path + 'hog_dt_pca1.model'
hog_dt_pca_model2 = hog_dt_path + 'hog_dt_pca2.model'


def run_decision_tree(xtrain, ytrain, xtest, ytest, model, retry=True):
    # Decision Tree train
    if (os.path.exists(model) == False or retry == True):
        print('---Decision Tree train start---')
        Decision_Tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2,
                                               min_samples_leaf=1,
                                               max_features=None, min_impurity_decrease=0)
        print(np.shape(np.asarray(xtrain)))
        Decision_Tree.fit(xtrain, ytrain)
        score = Decision_Tree.score(xtrain, ytrain)
        print('Decision Tree train score', round(score, 3))
        joblib.dump(Decision_Tree, model)
        print('---SDecision Tree train end---')
    else:
        print('--- Decision Tree train score loading ---')
        decision_tree_train = joblib.load(model)
        score = decision_tree_train.score(xtrain, ytrain)
        print('Decision Tree train score', round(score, 3))
        print('--- Decision Tree train score done ---')

    # SVM test
    decision_tree_test = joblib.load(model)
    print('---Decision Tree test start---')
    test_score = decision_tree_test.score(xtest, ytest)
    print('Decision Tree test score: ', round(test_score, 3))
    print('Decision Tree Confusion Metrics and Reports')
    decision_tree_pred = decision_tree_test.predict(xtest)
    cal_confusion(ytest, decision_tree_pred)
    print('---Decision Tree test end---')


print('--- Original Data ---')
run_decision_tree(train_gradient_list, train_labels, test_gradient_list, test_labels, hog_dt_model)
print()

print('--- Feature decomposed to 128 ---')
run_decision_tree(train_pca1, train_labels, test_pca1, test_labels, hog_dt_pca_model1)
print()

print('--- Feature decomposed to 512 ---')
run_decision_tree(train_pca2, train_labels, test_pca2, test_labels, hog_dt_pca_model2)
print()
