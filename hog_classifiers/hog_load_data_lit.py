import h5py
import os

def hog_load_data():
    trainData = './hog_data_lit/train.hdf5'
    testData = './hog_data_lit/test.hdf5'

    # Get original data
    if (os.path.exists(trainData)):
        with h5py.File(trainData, 'r') as f:
            print(f.keys())
            train_gradient_list = f.get('train_gradient_list')[:]
    else:
        print("Please run hog_get_data_lit.py first")

    if (os.path.exists(testData)):
        with h5py.File(testData, 'r') as f:
            print(f.keys())
            test_gradient_list = f.get('test_gradient_list')[:]
    else:
        print("Please run hog_get_data_lit.py first")

    return (train_gradient_list, test_gradient_list)