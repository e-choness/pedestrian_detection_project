import h5py
import os

def hog_load_data():
    trainData = './hog_data/train.hdf5'
    testData = './hog_data/test.hdf5'
    trainPCA1 = './hog_data/trainpca1.hdf5'
    testPCA1 = './hog_data/testpca1.hdf5'
    trainPCA2 = './hog_data/trainpca2.hdf5'
    testPCA2 = './hog_data/testpca2.hdf5'
    # Get original data
    if (os.path.exists(trainData)):
        with h5py.File(trainData, 'r') as f:
            print(f.keys())
            train_gradient_list = f.get('train_gradient_list')[:]
            train_labels = f.get('train_labels')[:]
    else:
        print("Please run hog_get_data.py first")

    if (os.path.exists(testData)):
        with h5py.File(testData, 'r') as f:
            print(f.keys())
            test_gradient_list = f.get('test_gradient_list')[:]
            test_labels = f.get('test_labels')[:]
    else:
        print("Please run hog_get_data.py first")

    # Get decomposed data
    if (os.path.exists(trainPCA1) and os.path.exists(testPCA1)):
        with h5py.File(trainPCA1, 'r') as f:
            print(f.keys())
            train_pca1 = f.get('train_pca1')[:]
        with h5py.File(testPCA1, 'r') as f:
            print(f.keys())
            test_pca1 = f.get('test_pca1')[:]
    else:
        print("Please run hog_get_data.py first")

    if (os.path.exists(trainPCA2) and os.path.exists(testPCA2)):
        with h5py.File(trainPCA2, 'r') as f:
            print(f.keys())
            train_pca2 = f.get('train_pca2')[:]
        with h5py.File(testPCA2, 'r') as f:
            print(f.keys())
            test_pca2 = f.get('test_pca2')[:]
    else:
        print("Please run hog_get_data.py first")

    return (train_gradient_list, train_labels), (test_gradient_list, test_labels), (train_pca1, test_pca1), (train_pca2, test_pca2)