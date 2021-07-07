import numpy as np
import joblib
import os

from sklearn.tree import DecisionTreeClassifier

from haar_load_data import haar_load_data
from haar_confusion_metrics import cal_confusion

(train_data_list, train_labels), (test_data_list, test_labels) = haar_load_data()

haar_dt_path = './haar_models/decision_tree/'
if (os.path.exists(haar_dt_path) == False):
    os.mkdir(haar_dt_path)

# Model path
haar_dt_model = haar_dt_path + 'haar_dt.model'


def run_decision_tree(xtrain, ytrain, xtest, ytest, model, retry=True):
    # Decision Tree train
    if (os.path.exists(model) == False or retry==True):
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
    dt_pred = decision_tree_test.predict(xtest)
    cal_confusion(ytest, dt_pred)
    print('---Decision Tree test end---')


print('--- Original Data ---')
run_decision_tree(train_data_list, train_labels, test_data_list, test_labels, haar_dt_model)
print()


