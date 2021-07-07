import numpy as np
import pickle
import os
import time
from sklearn.svm import LinearSVC
import joblib
import cv2


winSize = (8, 8)
blockSize = (4, 4)
blockStride = (4, 4)
cellSize = (2, 2)
nbins = 9
CIFAR = r'C:\Users\Administrator\Downloads\cifar-10-batches-py'
os.listdir(CIFAR)

with open(os.path.join(CIFAR,'data_batch_1') , 'rb') as f:
    data = pickle.load(f,encoding='bytes')


def load_data(filename):
    """read data from data file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


class CifarData:
    def __init__(self, filenames):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)

        self._newdata = np.zeros([len(self._data), 32, 32, 3])
        for i in range(len(self._data)):
            image_arr = self._data[i].reshape((3, 32, 32))
            image_arr = image_arr.transpose((1, 2, 0))
            self._newdata[i] = image_arr

        self._labels = np.hstack(all_labels)

    def get_data(self):
        return self._newdata, self._labels


train_filenames = [os.path.join(CIFAR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR, 'test_batch')]

train_data = CifarData(train_filenames)
test_data = CifarData(test_filenames)
X_train, y_train = train_data.get_data()
X_test, y_test = test_data.get_data()
print(X_train.shape)
print(X_test.shape)
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
winStride = (4, 4)
padding = (1, 1)

new_x_train = np.ones([50000, 11664])
for i, d in enumerate(X_train):
    d = np.uint8(d)
    train_hog = hog.compute(d, winStride, padding).reshape((-1,))
    new_x_train[i] = train_hog
print(new_x_train.shape)
new_x_test = np.ones([10000, 11664])
for i, d in enumerate(X_test):
    d = np.uint8(d)
    test_hog = hog.compute(d, winStride, padding).reshape((-1,))
    new_x_test[i] = test_hog
print(new_x_test.shape)

t0 = time.time()
SVM = LinearSVC(C=1)
SVM.fit(new_x_train, y_train)
joblib.dump(SVM, './hog_svm.model')

svm = joblib.load('./hog_svm.model')

t1 = time.time()
print("fit time", t1- t0)
print(svm.score(new_x_test, y_test))
print("score time:", time.time()-t1)


# new_x_train = np.ones([500, 32*32])
# for i, d in enumerate(X_train):
#     d = np.uint8(d)
#     train_hog = hog.compute(d, winStride, padding).reshape((-1,))
#     new_x_train[i] = train_hog
# print(new_x_train.shape)
# new_x_test = np.ones([100, 11664])
# for i, d in enumerate(X_test):
#     d = np.uint8(d)
#     test_hog = hog.compute(d, winStride, padding).reshape((-1,))
#     new_x_test[i] = test_hog
# print(new_x_test.shape)
#
# t0 = time.time()
# SVM = LinearSVC(C=1)
# SVM.fit(new_x_train, y_train)
# joblib.dump(SVM, './hog_svm.model')
#
# svm = joblib.load('./hog_svm.model')
#
# t1 = time.time()
# print("fit time", t1- t0)
# print(svm.score(new_x_test, y_test))


