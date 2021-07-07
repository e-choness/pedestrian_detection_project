import cv2
import random
import h5py
import os
from PIL import Image
from sklearn.decomposition import PCA

pos_images_path = r'C:/Users/Administrator/Desktop/INRIAPerson/train_64x128_H96/pos.lst'
neg_images_path = r'C:/Users/Administrator/Desktop/INRIAPerson/Train/neg.lst'

test_pos_images_path = r'C:/Users/Administrator/Desktop/INRIAPerson/test_64x128_H96/pos.lst'
test_neg_images_path = r'C:/Users/Administrator/Desktop/INRIAPerson/Test/neg.lst'

data_path = './hog_data/'
if (os.path.exists(data_path) == False):
    os.mkdir(data_path)

trainData = data_path + 'train1.hdf5'
testData = data_path + 'test1.hdf5'
trainPCA1 = data_path + 'trainpca1.hdf5'
testPCA1 = data_path + 'testpca1.hdf5'
trainPCA2 = data_path + 'trainpca2.hdf5'
testPCA2 = data_path + 'testpca2.hdf5'

n1 = 128
n2 = 512

def load_images(dir):
    img_list = []
    file = open(dir)
    img_name = file.readline()
    print('path ', dir)
    while img_name != '':  # end of file
        img_name = dir.rsplit(r'/', 1)[0] + r'/' + img_name.split('/', 1)[1].strip('\n')
        try:
            img = Image.open(img_name)
            img.save(img_name)
            img_list.append(cv2.imread(img_name))
        except:
            print('file not exist')
        img_name = file.readline()

    print('img-list ', len(img_list))
    return img_list


# 从没有人的原始图片中随机裁出2张64*128的图片作为负样本
def sample_neg(full_neg_list, neg_list, size, save=None):
    random.seed(1)
    width, height = size[1], size[0]
    for i in range(len(full_neg_list)):
        for j in range(2):
            y = int(random.random() * (len(full_neg_list[i]) - height))
            x = int(random.random() * (len(full_neg_list[i][0]) - width))
            img = full_neg_list[i][y:y + height, x:x + width]
            print(save, i*2 + j)
            neg_list.append(img)
            if save:
                dest = save + str(i*2 + j) + '.png'
                cv2.imwrite(dest, img)
    return neg_list


# wsize: 处理图片大小，通常是64*128, 输入图片尺寸>=wsize
def computeHOGs(img_list, gradient_list, wsize=(128, 64)):
    hog = cv2.HOGDescriptor()
    # hog.winSize = wsize
    for i in range(len(img_list)):
        if img_list[i].shape[1] >= wsize[1] and img_list[i].shape[0] >= wsize[0]:
            roi = img_list[i][(img_list[i].shape[0] - wsize[0]) // 2: (img_list[i].shape[0] - wsize[0]) // 2 + wsize[0], \
                  (img_list[i].shape[1] - wsize[1]) // 2: (img_list[i].shape[1] - wsize[1]) // 2 + wsize[1]]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gradient_list.append(hog.compute(gray).reshape((-1,)))
    return gradient_list


def prepareData(pos_images_path, neg_images_path, save=None):
    neg_list = []
    gradient_list = []
    labels = []
    pos_list = load_images(pos_images_path)
    full_neg_list = load_images(neg_images_path)
    sample_neg(full_neg_list, neg_list, [128, 64], save)
    computeHOGs(pos_list, gradient_list)
    [labels.append(1) for _ in range(len(pos_list))]
    computeHOGs(neg_list, gradient_list)
    [labels.append(-1) for _ in range(len(neg_list))]
    return gradient_list, labels

def data_pca(x_train, x_test, n_components):
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, random_state=42).fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_test_pca

# Original Data
if (os.path.exists(trainData)==False):
    train_gradient_list, train_labels = prepareData(pos_images_path, neg_images_path, save='INRIAPerson/train_haar/neg')
    with h5py.File(trainData, 'w') as f:
        f.create_dataset('train_gradient_list', data=train_gradient_list, compression='gzip', compression_opts=5)
        f.create_dataset('train_labels', data=train_labels, compression='gzip', compression_opts=5)
else:
    with h5py.File(trainData, 'r') as f:
        print(f.keys())
        train_gradient_list = f.get('train_gradient_list')[:]
        train_labels = f.get('train_labels')[:]
    print("train.hdf5 already exits.")

if (os.path.exists(testData)==False):
    test_gradient_list, test_labels = prepareData(test_pos_images_path, test_neg_images_path, save='INRIAPerson/test_haar/neg')
    with h5py.File(testData, 'w') as f:
        f.create_dataset('test_gradient_list', data=test_gradient_list, compression='gzip', compression_opts=5)
        f.create_dataset('test_labels', data=test_labels, compression='gzip', compression_opts=5)
else:
    with h5py.File(testData, 'r') as f:
        print(f.keys())
        test_gradient_list = f.get('test_gradient_list')[:]
        test_labels = f.get('test_labels')[:]
    print("test.hdf5 already exits.")

# Data Decomposition
if (os.path.exists(trainPCA1)==False and os.path.exists(testPCA1)==False):
    train_pca1, test_pca1 = data_pca(train_gradient_list, test_gradient_list, n1)
    with h5py.File(trainPCA1, 'w') as f:
        f.create_dataset('train_pca1', data=train_pca1, compression='gzip', compression_opts=5)
    with h5py.File(testPCA1, 'w') as f:
        f.create_dataset('test_pca1', data=test_pca1, compression='gzip', compression_opts=5)
else:
    print("trainpca1.hdf5 and testpca1.hdf5 already exits.")

if(os.path.exists(trainPCA2)==False and os.path.exists(testPCA2)==False):
    train_pca2, test_pca2 = data_pca(train_gradient_list, test_gradient_list, n2)
    with h5py.File(trainPCA2, 'w') as f:
        f.create_dataset('train_pca2', data=train_pca2, compression='gzip', compression_opts=5)
    with h5py.File(testPCA2, 'w') as f:
        f.create_dataset('test_pca2', data=test_pca2, compression='gzip', compression_opts=5)
else:
    print("trainpca2.hdf5 and testpca2.hdf5 already exits.")

