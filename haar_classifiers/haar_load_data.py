import pickle
import numpy as np
import os

def haar_load_data():
    trainData = './haar_data/informed_train_data.p'
    testData = './haar_data/informed_test_data.p'

    # Get original data
    if (os.path.exists(trainData)):

        informed_train = pickle.load(open(trainData, 'rb'))
        train_data_list, train_labels = informed_train['input'], informed_train['labels']
    else:
        print("Please run haar_get_data.py first")
        return

    if (os.path.exists(testData)):
        informed_test = pickle.load(open(testData, 'rb'))
        test_data_list, test_labels = informed_test['input'], informed_test['labels']
    else:
        print("Please run haar_get_data.py first")
        return

    print(train_data_list.shape, len(train_labels))
    print(test_data_list.shape, len(test_labels))

    train_data_list = np.concatenate([train_data_list[0:2000] , train_data_list[2414:]])
    train_labels[2000:] = -1
    train_labels = np.concatenate([train_labels[0:2000], train_labels[2414:]])


    test_data_list = np.concatenate([test_data_list[0:] , test_data_list[-6:]])
    test_labels[1132:] = -1
    test_labels = np.concatenate([test_labels[0:], test_labels[-6:]])
    return (train_data_list, train_labels), (test_data_list, test_labels)

(train_data_list, train_labels), (test_data_list, test_labels) = haar_load_data()

print(train_data_list.shape, len(train_labels))
print(train_labels)
print(test_data_list.shape, len(test_labels))
print(test_labels)