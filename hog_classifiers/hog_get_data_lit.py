import cv2
import random
import h5py
import os
from PIL import Image
from sklearn.decomposition import PCA

train_path = r'E:/PycharmProjects/pythonProject/datasets/INRIAPerson_lit/train_64x128_H96/pos/'
test_path = r'E:/PycharmProjects/pythonProject/datasets/INRIAPerson_lit/test_64x128_H96/pos/'

data_path = './hog_data_lit/'
if (os.path.exists(data_path) == False):
    os.mkdir(data_path)
model_path = './hog_models/'
if (os.path.exists(model_path) == False):
    os.mkdir(model_path)

trainData = data_path + 'train.hdf5'
testData = data_path + 'test.hdf5'

def get_image_list(root_path):
    complete_paths = []
    image_names = os.listdir(root_path)
    for i in image_names:
        path = os.path.join(root_path, i)
        complete_paths.append(path)

    return complete_paths

# wsize: 处理图片大小，通常是64*128, 输入图片尺寸>=wsize
def computeHOGs(img_list, gradient_list, wsize=(128, 64)):
    hog = cv2.HOGDescriptor()
    # hog.winSize = wsize
    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        if img.shape[1] >= wsize[1] and img.shape[0] >= wsize[0]:
            roi = img[(img.shape[0] - wsize[0]) // 2: (img.shape[0] - wsize[0]) // 2 + wsize[0], \
                  (img.shape[1] - wsize[1]) // 2: (img.shape[1] - wsize[1]) // 2 + wsize[1]]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gradient_list.append(hog.compute(gray).reshape((-1,)))
    return gradient_list


def prepareData(image_root):
    gradient_list = []
    images_path = get_image_list(image_root)
    # pos_list = load_images(images_path)
    computeHOGs(images_path, gradient_list)
    return gradient_list

def data_pca(x_train, x_test, n_components):
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, random_state=42).fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_test_pca

retry = True
# Original Data
if (os.path.exists(trainData)==False or retry==True):
    train_gradient_list = prepareData(train_path)

    with h5py.File(trainData, 'w') as f:
        f.create_dataset('train_gradient_list', data=train_gradient_list, compression='gzip', compression_opts=5)
    #    f.create_dataset('train_labels', data=train_labels, compression='gzip', compression_opts=5)
else:
    with h5py.File(trainData, 'r') as f:
        print(f.keys())
        train_gradient_list = f.get('train_gradient_list')[:]
    print("train.hdf5 already exits.")

if (os.path.exists(testData)==False or retry==True):
    test_gradient_list = prepareData(test_path)
    with h5py.File(testData, 'w') as f:
        f.create_dataset('test_gradient_list', data=test_gradient_list, compression='gzip', compression_opts=5)
else:
    with h5py.File(testData, 'r') as f:
        print(f.keys())
        test_gradient_list = f.get('test_gradient_list')[:]
    print("test.hdf5 already exits.")
