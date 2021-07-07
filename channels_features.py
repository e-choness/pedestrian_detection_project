import cv2
import numpy as np
from math import trunc
import os

class FeatureGenerator:
    def __init__(self, templates):
        """
            Instantiates feature generator with lists to store feature vectors and corresponding template
            information for each feature vector
        """

        self.templates = templates
        self.feature_info = []

    def generate_features(self, cfeats):
        """ Generates feature vectors associated with each template """

        self.features = []
        self.feature_info = []

        _, _, k_channels = cfeats.shape

        for indx, t in enumerate(self.templates):

            #=====[ Get channel and template from (channel, template) tuple t
            k = t[1]
            t = t[0]

            x, y, size, W = t
            w, h = size

            cell_feats = np.copy(cfeats[y:y + h, x:x + w, k])
            #=====[ Multiply channel features by template weight matrix W and sum ]=====
            self.features.append(np.sum(np.multiply(cell_feats, W)))

        return self.features

class ChannelFeatures:
    CELL_SIZE = 6  # pixels
    IMG_HEIGHT = 128
    IMG_WIDTH = 64
    H_CELLS = trunc(IMG_HEIGHT/CELL_SIZE)
    W_CELLS = trunc(IMG_WIDTH/CELL_SIZE)
    N_CHANNELS = 11
    NUM_HOG_BINS = 6

    def __init__(self):
        pass

    def _pool(self, vol, H_cells, W_cells):
        _, _, depth = vol.shape
        feats = np.zeros((H_cells, W_cells, depth))
        for i in range(H_cells-1):
            h_offset = i*self.CELL_SIZE
            for j in range(W_cells-1):
                w_offset = j*self.CELL_SIZE
                subluv = vol[h_offset:h_offset + self.CELL_SIZE, w_offset:w_offset + self.CELL_SIZE, :]
                feats[i, j, :] = np.sum(subluv)
        return feats

    def _compute_luv(self, img, H_cells, W_cells):
        luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        feats = self._pool(luv, H_cells, W_cells)
        return feats

    def _compute_gradients(self, img, H_cells, W_cells):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        sobelx64f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        abs_sobel64f = np.absolute(sobelx64f)
        sobel_x = np.uint8(abs_sobel64f)
        sobel_x = self._pool(sobel_x.reshape(H, W, 1), H_cells, W_cells)

        sobely64f = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        abs_sobel64f = np.absolute(sobely64f)
        sobel_y = np.uint8(abs_sobel64f)
        sobel_y = self._pool(sobel_y.reshape(H, W, 1), H_cells, W_cells)
        return np.dstack((sobel_x, sobel_y))

    def _compute_hog(self, img, H_cells, W_cells):
        winSize = (W_cells*self.CELL_SIZE, H_cells*self.CELL_SIZE)
        blockSize = (self.CELL_SIZE, self.CELL_SIZE)
        blockStride = (self.CELL_SIZE, self.CELL_SIZE)
        cellSize = (self.CELL_SIZE, self.CELL_SIZE)
        nbins = self.NUM_HOG_BINS
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        hog_feats = hog.compute(img)
        return hog_feats.reshape(H_cells, W_cells, self.NUM_HOG_BINS)

    def compute_channels(self, img, resize=False):

        if resize:
            img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
        img_H, img_W, _ = img.shape
        H_cells = trunc(img_H/self.CELL_SIZE)
        W_cells = trunc(img_W/self.CELL_SIZE)
        img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=.87)

        # LUV channels
        luv = self._compute_luv(img, H_cells, W_cells)

        # Gradient magnitudes in X and Y (Sobel filters)
        grads = self._compute_gradients(img, H_cells, W_cells)

        # Gradient histogram channels (HOG, 6 bins)
        hog = self._compute_hog(img, H_cells, W_cells)

        channels = np.dstack((luv, grads, hog))
        return channels


class TemplateGenerator():

    def __init__(self, shape_model=None, cell_size=None):
        """ Instantiates TempalteGenerator. Creates default shape model if none provided"""

        if not shape_model:

            # Number of pixels in each cell
            self.cell_size = 6

            # Make dummy data shape model
            shape_model = np.zeros([20, 10])
            shape_model[2:4, 4:6] = 1
            shape_model[4:11, 2:8] = 2
            shape_model[11:18, 3:7] = 3
            self.shape_model = shape_model

        else:

            self.cell_size = cell_size
            self.shape_model = shape_model

    def generate_sizes(self, w_max=4, h_max=3):
        """ Generates set of possible template sizes. """

        # Define width and height constraints in terms of cells
        w_vals = range(1, w_max + 1)
        h_vals = range(1, h_max + 1)

        # Generate size pool for template sizes
        sizes = [(w, h) for w in w_vals for h in h_vals]
        self.sizes = sizes[1:]

    def generate_templates(self):
        """
            Generates templates by convolving windows defined by sizes over the shape model.

            Template Format:

                        (x, y, size, W)

                        x: x position of upper left hand corner of template (in terms of cells)
                        y: y position of upper left hand corner of template (in terms of cells)
                        size: (width, height) of template (in terms of cells)
                        W: weight matrix (weights for each cell)
        """

        templates = []
        cell_size = self.cell_size

        # Slide each size template over the entire shape model and generate templates
        for size in self.sizes:
            w = size[0]
            h = size[1]

            # Slide template with dimenions specified by size across the entire shape model
            for y in range(self.shape_model.shape[0] - h):
                for x in range(self.shape_model.shape[1] - w):

                    mat_temp = np.copy(self.shape_model[y:y + h, x:x + w])
                    unique = np.unique(mat_temp)

                    # Check to make sure template holds some shape model information
                    if len(unique) > 1:

                        # Binary template: set values to 1 and 0 and add template
                        if len(unique) == 2:
                            idx1 = mat_temp == unique[0]
                            idx2 = mat_temp == unique[1]

                            mat_temp[idx1] = 1
                            mat_temp[idx2] = 0
                            templates.append((x, y, size, mat_temp))

                        # Ternary template: set values to -1, 0, 1 -- add template -- repeat with all permutations
                        else:
                            # Get unique value indices
                            idx1 = mat_temp == unique[0]
                            idx2 = mat_temp == unique[1]
                            idx3 = mat_temp == unique[2]

                            mat_temp[idx1] = -1
                            mat_temp[idx2] = 0
                            mat_temp[idx3] = 1
                            templates.append((x, y, size, mat_temp))

                            mat_temp[idx1] = 1
                            mat_temp[idx2] = -1
                            mat_temp[idx3] = 0
                            templates.append((x, y, size, mat_temp))

                            mat_temp[idx1] = 0
                            mat_temp[idx2] = 1
                            mat_temp[idx3] = -1
                            templates.append((x, y, size, mat_temp))

        self.templates = np.asarray(templates, dtype=object)
        self.remove_duplicates()
        self.shift_templates()
        self.normalize_templates()

        print('Created %d templates' % (len(self.templates)))
        return self.templates

    def remove_duplicates(self):
        """ Removes all duplicate templates """

        to_remove = []

        # Compare every template against each other
        for idx, t1 in enumerate(self.templates):
            for idx2, t2 in enumerate(self.templates[idx + 1:]):

                # If templates at the same x,y coordinate
                if t1[0] == t2[0] and t1[1] == t2[1]:
                    _, _, size1, W1 = t1
                    _, _, size2, W2 = t2
                    w1, h1 = size1
                    w2, h2 = size2
                    wmax = max([w1, w2])
                    hmax = max([h1, h2])

                    # Expand matrices
                    W1p = np.zeros([hmax, wmax])
                    W2p = np.zeros([hmax, wmax])
                    W1p[:h1, :w1] = W1
                    W2p[:h2, :w2] = W2

                    # If matrices subtracted from each other == 0, remove one
                    if np.sum(np.abs(W1p - W2p)) == 0:
                        to_remove.append(idx)
                        break

        # Get indices for subset of templates
        indices = [x for x in range(len(self.templates)) if x not in to_remove]
        self.templates = self.templates[indices]

    def shift_templates(self):

        new_templates = []

        # Iterate through each template and add new template/shift up, down, left, right one cell if possible.
        for t in self.templates:
            x, y, size, W = t

            if y < self.shape_model.shape[0] - 1:
                new_templates.append((x, y + 1, size, W))

            if y > 0:
                new_templates.append((x, y - 1, size, W))

            if x < self.shape_model.shape[1] - 1:
                new_templates.append((x + 1, y, size, W))

            if x > 0:
                new_templates.append((x - 1, y, size, W))

        new_templates = np.asarray(new_templates, dtype=object)

        self.templates = np.concatenate((self.templates, new_templates), axis=0)

    def normalize_templates(self):

        for idx, t in enumerate(self.templates):

            x, y, size, W = t

            W1 = np.copy(W)
            W2 = np.copy(W)

            W1[W1 != 1] = 0
            W2[W2 != -1] = 0

            s1 = np.sum(W1)
            s2 = np.sum(-W2)

            if s2:
                self.templates[idx] = (x, y, size, np.copy(W1 / s1 + W2 / s2))
            else:
                self.templates[idx] = (x, y, size, np.copy(W1 / s1))


def _get_feature_matrix(cf, fg, X, images, offset=0):
    """ Append feature vectors for each training example in images to X """
    # =====[ Iterate through images and calculate feature vector for each ]=====
    print(len(images))
    for idx, img in enumerate(images):

        # try:
        print('img loaded', img)
        # print(os.path.exists(img))
        # print(os.path.isfile(img))
        # cvimg = cv2.imread(img)
        # print('----: ', type(cvimg))
        cfeats = cf.compute_channels(cv2.imread(img))
        feature_vec = fg.generate_features(cfeats)

        # =====[ Add feature vector to input matrix ]=====
        X[idx + offset, :] = feature_vec

        # except Exception as e:
        #     print('Could not add image at index: ', idx + offset)

    return X


def _make_labels(n_pos, n_neg):
    """ Takes number of positive and negative images and returns appropriate label vector """

    Y = np.zeros((n_pos + n_neg))
    Y[:n_pos] = 1

    return Y


def _get_image_paths(base_dir, pos_filename, neg_filename):
    """ Get list of image paths in base_dir from each file_name """

    with open(os.path.join(base_dir, pos_filename)) as f:
        pos_list = f.readlines()
        pos_list = [base_dir + '/pos/' + x.strip() for x in pos_list]
    with open(os.path.join(base_dir, neg_filename)) as f:
        neg_list = f.readlines()
        neg_list = [base_dir + '/neg/' + x.strip() for x in neg_list]

    print('-----> Loaded {} positive image paths and {} negative image paths'.format(str(len(pos_list)),
                                                                                     str(len(neg_list))))
    return pos_list, neg_list

