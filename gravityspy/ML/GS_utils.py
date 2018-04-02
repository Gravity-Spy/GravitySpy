
""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.transform import resize
import os
import numpy as np
import gzip
import time

import numpy
import gzip
import pickle

from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from datetime import datetime
from datetime import datetime
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.transform import resize
import os
import random, shutil
import numpy as np
import gzip
import time
import sys
import matplotlib.pyplot as plt
from keras import backend as K
from keras.regularizers import l2
import numpy
import gzip
import pickle


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def concatenate_views(image_set1, image_set2,image_set3, image_set4, image_size):
    img_rows = image_size[0]
    img_cols = image_size[1]

    assert len(image_set1) == len(image_set2)
    concat_images_set = np.zeros((len(image_set1), 1, img_rows * 2, img_cols))
    for i in range(0, len(image_set1)):
        concat_images_set[i, :, :, :] = np.append(image_set1[i, :, :, :], image_set2[i, :, :, :], axis=1)

    assert len(image_set3) == len(image_set4)
    concat_images_set2 = np.zeros((len(image_set3), 1, img_rows * 2, img_cols))
    for i in range(0, len(image_set3)):
        concat_images_set2[i, :, :, :] = np.append(image_set3[i, :, :, :], image_set4[i, :, :, :], axis=1)

    out = np.append(concat_images_set,concat_images_set2, axis=3)
    return out


#4/2/2018
def create_model_folder(file_name, save_adr, verbose):
    folder_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    print ("storing results in folder: " + folder_name)

    full_adr = save_adr + folder_name
    if not os.path.exists(full_adr):
        os.mkdir(full_adr)
        print ('making full adress: '+full_adr)
    if verbose:
        shutil.copy2(file_name, full_adr + '/')
        shutil.copy2('GS_utils.py', full_adr + '/')
        #shutil.copy2('glitch_dist_utils.py', full_adr + '/')
    return full_adr

def listdir_nohidden(path):

    for f in os.listdir(path):

        if not f.startswith('.'):

            yield f

def my_load_dataset(dataset):
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    (train_set_x, train_set_y, train_set_name ) = train_set
    (valid_set_x, valid_set_y, valid_set_name) = valid_set
    (test_set_x, test_set_y, test_set_name) = test_set
    rval = [(train_set_x, train_set_y, train_set_name), (valid_set_x, valid_set_y, valid_set_name), (test_set_x, test_set_y, test_set_name)]
    return rval

def load_dataset_unlabelled_glitches(dataset,verbose):
    with gzip.open(dataset, 'rb') as f:
        try:
            test_set = pickle.load(f, encoding='latin1')
        except:
            test_set = pickle.load(f)
    if verbose:
        print ('my_load_dataset_test size test set shape', len(test_set))
    #[[(test_set_x, test_set_y, test_set_name)] = test_set
    #rval = [(test_set_x, test_set_y, test_set_name)]
    return test_set

def my_save_dataset_test(save_adress, imgs, labels, names, str_name):
    #data_size = imgs.shape[0]
    "Partitioning into train, validation and test sets"
    #trainNumber = data_size * 0.75
    #validNumber = data_size * 0.125
    #testNumber = data_size

    #trainset, valisdet, testset = numpy.split(new_ind, [trainNumber + 1, trainNumber + validNumber + 1])

    labels = numpy.array(labels)
    names = numpy.array(names)

    ## Gray Images
    '''input_tr = imgs[trainset]
    target_tr = labels[trainset]
    names_tr = names[trainset]

    input_val = imgs[valisdet]
    target_val = labels[valisdet]
    names_val = names[valisdet]'''

    input_te = imgs
    target_te = labels
    names_te = names

    out = [[input_te, target_te, names_te]]

    ## Normalized between 0 and 1 /255.0
    div_norm = 255.0
    out = [numpy.divide(input_te, div_norm), target_te, names_te]
    date = time.strftime("%x")

    f = gzip.open(save_adress + 'img' +str_name + 'class' + str(max(labels)+1) + '_norm.pkl.gz', 'wb')
    #f = gzip.open('img_.pkl.gz', 'wb')
    pickle.dump(out, f, protocol=2)
    f.close()


def my_save_dataset(save_adress, new_ind, imgs, labels, names, str_name ):
    data_size = imgs.shape[0]
    "Partitioning into train, validation and test sets"
    trainNumber = int(data_size * 0.75)
    validNumber = int(data_size * 0.125)
    testNumber = int(data_size * 0.125)

    trainset, valisdet, testset = numpy.split(new_ind, [trainNumber + 1, trainNumber + validNumber + 1])

    labels = numpy.array(labels)
    names = numpy.array(names)

    ## Gray Images
    input_tr = imgs[trainset]
    target_tr = labels[trainset]
    names_tr = names[trainset]

    input_val = imgs[valisdet]
    target_val = labels[valisdet]
    names_val = names[valisdet]

    input_te = imgs[testset]
    target_te = labels[testset]
    names_te = names[testset]

    out = [[input_tr, target_tr, names_tr], [input_val, target_val, names_val], [input_te, target_te, names_te]]

    ## Normalized between 0 and 1 /255.0
    div_norm = 255.0
    out = [[numpy.divide(input_tr, div_norm), target_tr, names_tr], [numpy.divide(input_val, div_norm), target_val, names_val],
           [numpy.divide(input_te, div_norm), target_te, names_te]]
    date = time.strftime("%x")

    f = gzip.open(save_adress + 'img' +str_name + 'class' + str(max(labels)+1) + '_norm.pkl.gz', 'wb')
    #f = gzip.open('img_.pkl.gz', 'wb')
    pickle.dump(out, f, protocol=2)
    f.close()


def my_read_image(data_path, dataset_str):

    classes = os.listdir(data_path)
    classes = [c for c in classes if not c.startswith('.') ]
    classes = sorted(classes)

    imgs = []
    labels = []
    names = []

    for index, item in enumerate(classes):
      path = data_path + classes[index]
      #if os.path.isdir(path):
      if 1:
        print(path)
        files = os.listdir(path)
        files = [f for f in files if not f.startswith('.')]
        #np.savetxt('/Users/Sara/Desktop/data/pkl/all/path_files_' + dataset_str + '.csv', files, delimiter=',', fmt='%s')
        for f in files:
            if dataset_str in f:
                test = io.imread(os.path.join(path, f))
                test = test[66:532, 105:671, :]
               # plt.imshow(test)
                test = rgb2gray(test)
                test = rescale(test, [0.1, 0.1], mode='constant')
                # test = resize(test, (50, 50))
                #num_feature = test.shape
                dim = numpy.int(reduce(lambda x, y: x * y, test.shape))
                imgs.append(numpy.reshape(test, (dim)))  # vectorizing
                names.append(f[0:13])  # its name
                labels.append(index)
               # labels.append(1)
    np.savetxt('class2idx.csv', classes, delimiter=',', fmt='%s')
    return [imgs, labels, names]

def my_read_image2(data_path, dataset_str):

    classes = os.listdir(data_path)
    classes = [c for c in classes if os.path.isdir(data_path + c)]
    classes = sorted(classes)

    imgs = []
    labels = []
    names = []

    # np.savetxt('/Users/Sara/Desktop/data/pkl/all25/path_folders_'+ dataset_str +'.csv', classes, delimiter=',', fmt='%s')
    for index, item in enumerate(classes):
        path = data_path + classes[index]
        # if os.path.isdir(path):
        if 1:
            print(path)
            files = os.listdir(path)
            files = [f for f in files if not f.startswith('.')]
            # np.savetxt('/Users/Sara/Desktop/data/pkl/all/path_files_' + dataset_str + '.csv', files, delimiter=',', fmt='%s')
            for f in files:
                if dataset_str in f:
                    test = io.imread(os.path.join(path, f))
                    test = test[66:532, 105:671, :]
                    # plt.imshow(test)
                    test = rgb2gray(test)
                    test = rescale(test, [0.1, 0.1], mode='constant')
                    # test = resize(test, (50, 50))
                    # num_feature = test.shape
                    dim = numpy.int(reduce(lambda x, y: x * y, test.shape))
                    imgs.append(numpy.reshape(test, (dim)))  # vectorizing
                    names.append(f[0:13])  # its name
                    labels.append(index)
                    # labels.append(1)
    np.savetxt('class2idx.csv', classes, delimiter=',', fmt='%s')
    return [imgs, labels, names, classes]

def load_data(dataset):
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    (train_set_x, train_set_y) = train_set
    (valid_set_x, valid_set_y) = valid_set
    (test_set_x, test_set_y) = test_set
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

#4/2/2018
def build_cnn(img_rows, img_cols):

    W_reg = 1e-4
    print('regularization parameter: ', W_reg)
    model = Sequential()
    model.add(Conv2D(16, (5, 5), padding='valid', input_shape=(1, img_rows, img_cols), kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(32, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(W_reg)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    print (model.summary())
    return model

'''def build_cnn(img_rows, img_cols):
    model = Sequential()
    #model.add(Convolution2D(nb_filter=100, nb_row=5, nb_col=5,
    #model.add(Convolution2D(nb_filter=100, nb_row=5, nb_col=5,
    #                        init='glorot_uniform', activation='linear',
    #                        border_mode='valid',
    #                        input_shape=(1, img_rows, img_cols)))
    model.add(Conv2D(filters=100, kernel_size=(5, 5),
                            kernel_initializer='glorot_uniform', 
                            activation='linear',
                            padding='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Convolution2D(nb_filter=100, nb_row=5, nb_col=5,
    #model.add(Convolution2D(nb_filter=100, nb_row=5, nb_col=5,
    #                        init='glorot_uniform', activation='linear',
    #                        border_mode='valid'))
    model.add(Conv2D(filters=100, kernel_size=(5, 5),
                            kernel_initializer='glorot_uniform', 
                            activation='linear',
                            padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    return model'''


def build_partial_cnn1(img_rows, img_cols):
    model = Sequential()
    #model.add(Convolution2D(nb_filter=100, nb_row=5, nb_col=5,
    #model.add(Convolution2D(nb_filter=10, nb_row=2, nb_col=2,
    #                        init='glorot_uniform', activation='linear',
    #                        border_mode='valid',
    #                        input_shape=(1, img_rows, img_cols)))
    model.add(Conv2D(nb_filter=10, nb_row=2, nb_col=2,
                            init='glorot_uniform', activation='linear',
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))

    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Convolution2D(nb_filter=100, nb_row=5, nb_col=5,
    '''model.add(Convolution2D(nb_filter=512, nb_row=5, nb_col=5,
                            init='glorot_uniform', activation='linear',
                            border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))'''

    return model


def build_cnn_volume(img_rows, img_cols, n):
    model = Sequential()
    #model.add(Convolution2D(nb_filter=100, nb_row=5, nb_col=5,
    #                        init='glorot_uniform', activation='linear',
    #                        border_mode='valid',
    #                        input_shape=(n, img_rows, img_cols)))
    model.add(Conv2D(nb_filter=100, nb_row=5, nb_col=5,
                            init='glorot_uniform', activation='linear',
                            border_mode='valid',
                            input_shape=(n, img_rows, img_cols)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Convolution2D(nb_filter=100, nb_row=5, nb_col=5,
    #                        init='glorot_uniform', activation='linear',
    #                        border_mode='valid'))
    model.add(Conv2D(nb_filter=100, nb_row=5, nb_col=5,
                            init='glorot_uniform', activation='linear',
                            border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    return model


