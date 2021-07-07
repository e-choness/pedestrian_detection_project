import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from hog_load_data_lit import hog_load_data
from hog_get_data_lit import get_image_list

test_path = r'E:/PycharmProjects/pythonProject/datasets/INRIAPerson_lit/test_64x128_H96/pos/'

(train_gradient_list, test_gradient_list) = hog_load_data()

print(train_gradient_list.shape)
print(test_gradient_list.shape)

kmeans = KMeans(n_clusters=8, random_state=10)
kmeans.fit(train_gradient_list)

# print(kmeans.cluster_centers_)
ypred = kmeans.predict(test_gradient_list)
print(ypred)

result = ypred.tolist()
uniques = np.unique(ypred).tolist()
example_idx = []
for i in uniques:
    example_idx.append(result.index(i))

test_img_list = get_image_list(test_path)

fig, ax = plt.subplots(2, 4)
for i, axi in enumerate(ax.flat):
    id = example_idx[i]
    example_image = plt.imread(test_img_list[id])
    axi.imshow(example_image)
    axi.set(xticks=[], yticks=[])

plt.show()
