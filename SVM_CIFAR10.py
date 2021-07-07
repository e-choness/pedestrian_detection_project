import numpy as np
import pickle
import os
import time
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import joblib


CIFAR = r'C:\Users\Administrator\Downloads\cifar-10-batches-py'
os.listdir(CIFAR)
with open(os.path.join(CIFAR,'data_batch_1') , 'rb') as f:
    data = pickle.load(f,encoding='bytes')
    print(type(data))
    print(data.keys())

with open(os.path.join(CIFAR,'data_batch_1') , 'rb') as f:
    data = pickle.load(f,encoding='bytes')
    print(type(data[b'data']))
    print(type(data[b'batch_label']))
    print(type(data[b'labels']))
    print(type(data[b'filenames']))
    print(data[b'data'].shape)
    print(Counter(data[b'labels']))
    print(data[b'filenames'][1])
    print(data[b'data'][:2])
    print(data[b'labels'][:2])


def load_data(filename):
    """read data from data file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


# tensorflow.Dataset.
class CifarData:
    def __init__(self, filenames):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)

    def get_data(self):
        return self._data, self._labels


train_filenames = [os.path.join(CIFAR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR, 'test_batch')]

train_data = CifarData(train_filenames)
test_data = CifarData(test_filenames)

X_train, y_train = train_data.get_data()
X_test, y_test = test_data.get_data()
t1 = time.time()
print(X_train.shape)
pca = PCA(0.90)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
print(X_train_reduction.shape)
SVM = LinearSVC(C=1)
SVM.fit(X_train_reduction, y_train)
joblib.dump(SVM, './svm.model')

svm = joblib.load('./svm.model')

print(svm.score(X_test_reduction, y_test))
print("time:", time.time()-t1)

