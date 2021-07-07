from channels_features import *
import pickle


num_ft = 128
template_file = 'top_templates_1000.p'
templates = pickle.load(open(template_file, 'rb'), encoding='latin1')
templates = templates[:num_ft]
train_dir = r'INRIAPerson/train_haar'
test_dir = r'INRIAPerson/test_haar'

pos_path = r'pos.lst'
neg_path = r'neg.lst'


train_pos_images, train_neg_images = _get_image_paths(train_dir, pos_path, neg_path)
# train_neg_images = _get_image_paths(base_dir, train_neg_images_path)

test_pos_images, test_neg_images = _get_image_paths(test_dir, pos_path, neg_path)
# test_neg_images = _get_image_paths(base_dir, test_neg_images_path)

dump_train = 'informed_train_data.p'
dump_test = 'informed_test_data.p'

# print(os.path.exists(train_pos_images_path))
fg = FeatureGenerator(templates)
cf = ChannelFeatures()
print('templates generated: ', len(templates))

print('-----> Total training images to process: ', len(train_pos_images) + len(train_neg_images))
X_train = np.zeros((len(train_pos_images) + len(train_neg_images), len(templates)))

X_train = _get_feature_matrix(cf, fg, X_train, train_pos_images, 0)
X_train = _get_feature_matrix(cf, fg, X_train, train_neg_images, len(train_pos_images) - 1)
print('-----> Obtained feature matrix with shape {}'.format(str(X_train.shape)))


Y_train = _make_labels(len(train_pos_images), len(train_neg_images))

# =====[ If user specified a file name to save X, and Y to, pickle objects ]=====
if dump_train:
    pickle.dump({'input': X_train, 'labels': Y_train}, open(dump_train, 'wb'))
    print('-----> Successfully formulated and saved training X and Y')


print('-----> Total test images to process: ', len(test_pos_images) + len(test_neg_images))
X_test = np.zeros((len(test_pos_images) + len(test_neg_images), len(templates)))

X_test = _get_feature_matrix(cf, fg, X_test, test_pos_images, 0)
X_test = _get_feature_matrix(cf, fg, X_test, test_neg_images, len(test_pos_images) - 1)
print('-----> Obtained feature matrix with shape {}'.format(str(X_test.shape)))


Y_test = _make_labels(len(test_pos_images), len(test_neg_images))

# =====[ If user specified a file name to save X, and Y to, pickle objects ]=====
if dump_test:
    pickle.dump({'input': X_test, 'labels': Y_test}, open(dump_test, 'wb'))
    print('-----> Successfully formulated and saved X and Y')