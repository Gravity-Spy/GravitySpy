
""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""


from keras import backend as K
K.set_image_dim_ordering('th')
from datetime import datetime
import random, shutil

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rescale, resize
import os
import numpy as np

from keras.regularizers import l2
import numpy
import gzip, pickle
from keras.layers import Conv2D, MaxPooling2D, Flatten
from sklearn.cross_validation import StratifiedShuffleSplit
from functools import reduce

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Activation, Reshape


#4/2/2018
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


# 4/2/2018
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

# 4/3/2018
def load_data_single_channel(adress, img_spec):
    img_rows = img_spec[0]
    img_cols = img_spec[1]
    data = my_load_dataset(adress)
    test_set_x_1, test_set_y_1, test_set_name_1 = data[2]
    valid_set_x_1, valid_set_y_1, valid_set_name_1 = data[1]
    train_set_x_1, train_set_y_1, train_set_name_1 = data[0]
    test_set_x_1 = test_set_x_1.reshape(-1, 1, img_rows, img_cols)
    train_set_x_1 = train_set_x_1.reshape(-1, 1, img_rows, img_cols)
    valid_set_x_1 = valid_set_x_1.reshape(-1, 1, img_rows, img_cols)

    return [[train_set_x_1, train_set_y_1, train_set_name_1],
           [valid_set_x_1, valid_set_y_1, valid_set_name_1],
            [test_set_x_1, test_set_y_1, test_set_name_1]
           ]


#4/3/2018
def load_data_multiple_channel(ad, file_r, file_g, file_b, img_spec):
    [[tr_x_r, tr_y_r, tr_name_r], [val_x_r, val_y_r, val_name_r],
     [te_x_r, te_y_r, te_name_r]] = load_data_single_channel(ad + file_r, img_spec)
    [[tr_x_g, tr_y_g, tr_name_g], [val_x_g, val_y_g, val_name_g],
     [te_x_g, te_y_g, te_name_g]] = load_data_single_channel(ad + file_g, img_spec)
    [[tr_x_b, tr_y_b, tr_name_b], [val_x_b, val_y_b, val_name_b],
     [te_x_b, te_y_b, te_name_b]] = load_data_single_channel(ad + file_b, img_spec)
    assert np.all(tr_y_r == tr_y_g) and np.all(tr_y_g == tr_y_b)
    assert np.all(tr_name_r == tr_name_g) and np.all(tr_name_g == tr_name_b)

    assert np.all(val_y_r == val_y_g) and np.all(val_y_g == val_y_b)
    assert np.all(val_name_r == val_name_g) and np.all(val_name_g == val_name_b)

    assert np.all(te_y_r == te_y_g) and np.all(te_y_g == te_y_b)
    assert np.all(te_name_r == te_name_g) and np.all(te_name_g == te_name_b)

    return [np.concatenate((tr_x_r, tr_x_g, tr_x_b), axis=1), tr_y_r, tr_name_r], [
        np.concatenate((val_x_r, val_x_g, val_x_b), axis=1), val_y_r, val_name_r], [
               np.concatenate((te_x_r, te_x_g, te_x_b), axis=1), te_y_r, te_name_r]


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


def load_dataset_unlabelled_glitches(dataset):
    with gzip.open(dataset, 'rb') as f:
        try:
            test_set = pickle.load(f, encoding='latin1')
        except:
            test_set = pickle.load(f)
    print('reading pickle ...')
    return test_set


# 4/3/2018
def my_save_dataset_test(save_adress, imgs, labels, names, str_name):

    labels = numpy.array(labels)
    names = numpy.array(names)

    input_te = imgs
    target_te = labels
    names_te = names

    div_norm = 1.0
    out = [numpy.divide(input_te, div_norm), target_te, names_te]

    f = gzip.open(save_adress + 'img' +str_name + 'class' + str(max(labels)+1) + '_norm.pkl.gz', 'wb')
    pickle.dump(out, f, protocol=2)
    f.close()


# 4/3/2018
def my_save_dataset_stratified(save_adress, imgs, labels, names, str_name, rgb_flag):
    my_file2 = open(save_adress + 'dataset_info.txt', 'w')
    skf = StratifiedShuffleSplit(labels, 1, test_size=0.5, random_state=0)
   # skf = StratifiedShuffleSplit(labels, 1, test_size=0.3, random_state=0)


    for train_index, test_index in skf:
        trainset = train_index

    labels = numpy.array(labels)
    names = numpy.array(names)

    if rgb_flag:
        input_tr_1 = imgs[trainset, 0, :]
        input_tr_2 = imgs[trainset, 1, :]
        input_tr_3 = imgs[trainset, 2, :]
    else:
        input_tr = imgs[trainset]

    target_tr = labels[trainset]
    names_tr = names[trainset]
    my_file2.write('\n\n' +'Training set: \n')
    for name in names_tr:
        my_file2.write(str(name) + ',' + '\n')

    sub_labels = labels[test_index]
    sub_names = names[test_index]
    sub_imgs = imgs[test_index]

    skf2 = StratifiedShuffleSplit(sub_labels, 1, test_size=0.5, random_state=0)
    for testidx, valididx in skf2:
        valisdet = valididx
        testset = testidx

    if rgb_flag:
        input_val_1 = sub_imgs[valisdet, 0, :]
        input_val_2 = sub_imgs[valisdet, 1, :]
        input_val_3 = sub_imgs[valisdet, 2, :]

    else:
        input_val = sub_imgs[valisdet]

    target_val = sub_labels[valisdet]
    names_val = sub_names[valisdet]
    my_file2.write('\n\n' + 'Validation set: \n')
    for name in names_val:
        my_file2.write(str(name) + ',' + '\n')

    if rgb_flag:
        input_te_1 = sub_imgs[testset, 0, :]
        input_te_2 = sub_imgs[testset, 1, :]
        input_te_3 = sub_imgs[testset, 2, :]
    else:
        input_te = sub_imgs[testset]

    target_te = sub_labels[testset]
    names_te = sub_names[testset]
    my_file2.write('\n\n' + 'Test set: \n' + '\n')
    for name in names_te:
        my_file2.write(str(name) + ',' + '\n')

    my_file2.close()
    #out = [[input_tr, target_tr, names_tr], [input_val, target_val, names_val], [input_te, target_te, names_te]]

    ## Normalized between 0 and 1 /255.0
    #div_norm = 255.0
    div_norm = 1.0

    if rgb_flag:
        out1 = [[numpy.divide(input_tr_1, div_norm), target_tr, names_tr],
               [numpy.divide(input_val_1, div_norm), target_val, names_val],
               [numpy.divide(input_te_1, div_norm), target_te, names_te]]

        out2 = [[numpy.divide(input_tr_2, div_norm), target_tr, names_tr],
                [numpy.divide(input_val_2, div_norm), target_val, names_val],
                [numpy.divide(input_te_2, div_norm), target_te, names_te]]

        out3 = [[numpy.divide(input_tr_3, div_norm), target_tr, names_tr],
                [numpy.divide(input_val_3, div_norm), target_val, names_val],
                [numpy.divide(input_te_3, div_norm), target_te, names_te]]

        f1 = gzip.open(save_adress + 'img' + str_name + '_r_' + 'class' + str(max(labels)+1) + '_norm.pkl.gz', 'wb')
        pickle.dump(out1, f1, protocol=2)
        f1.close()

        f2 = gzip.open(save_adress + 'img' + str_name + '_g_' + 'class' + str(max(labels)+1) + '_norm.pkl.gz', 'wb')
        pickle.dump(out2, f2, protocol=2)
        f2.close()

        f3 = gzip.open(save_adress + 'img' + str_name + '_b_' + 'class' + str(max(labels)+1) + '_norm.pkl.gz', 'wb')
        pickle.dump(out3, f3, protocol=2)
        f3.close()

    else:
        out = [[numpy.divide(input_tr, div_norm), target_tr, names_tr], [numpy.divide(input_val, div_norm), target_val, names_val],
               [numpy.divide(input_te, div_norm), target_te, names_te]]

        f = gzip.open(save_adress + 'img' +str_name + 'class' + str(max(labels)+1) + '_norm.pkl.gz', 'wb')
        pickle.dump(out, f, protocol=2)
        f.close()
 

'''
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
    return [imgs, labels, names, classes]'''


# 4/3/2018
def pre_process_img(mat, resolution, rescale_flag):
    if rescale_flag:
        mat = rescale(mat, resolution, mode='constant', preserve_range='True')
    else:
        mat = resize(mat, (resolution, resolution), mode='constant', preserve_range='True')
    dim = numpy.int(reduce(lambda x, y: x * y, mat.shape))
    return numpy.reshape(mat, (dim))


def my_read_image2(save_address, data_path, dataset_str, resolution, for_similarity_measure_flag, rgb_flag, rescale_flag):

    classes = os.listdir(data_path)
    classes = [c for c in classes if not c.startswith('.')]
    if for_similarity_measure_flag:
        classes = [c for c in classes if not 'None_of_the_Above' in c]

    imgs = []
    labels = []
    names = []
    my_file = open(save_address + 'classes_distribution.txt', 'w')

    for index, item in enumerate(classes):
        counter = 0
        path = data_path + classes[index]
        print(path)

        files = os.listdir(path)
        files = [f for f in files if not f.startswith('.')]

        for f in files:
            if dataset_str in f:
                test = io.imread(os.path.join(path, f))
                test = test[66:532, 105:671, :3]
                # plt.imshow(test)
                if not rgb_flag:
                    test = rgb2gray(test)
                    imgs.append(pre_process_img(test, resolution, rescale_flag))
                else:
                    test1 = test[:, :, 0]
                    test1 = pre_process_img(test1, resolution, rescale_flag)
                    test2 = test[:, :, 1]
                    test2 = pre_process_img(test2, resolution, rescale_flag)
                    test3 = test[:, :, 2]
                    test3 = pre_process_img(test3, resolution, rescale_flag)
                    imgs.append([test1, test2, test3])

                names.append(f[0:13])  # glitch file's name
                labels.append(index)
                counter = counter + 1

        my_file.write(classes[index] + ': ' + str(counter)+'\n')

    my_file.close()
    np.savetxt(save_address + 'class2idx.csv', classes, delimiter=',', fmt='%s')
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
    w_reg = 1e-4
    print('regularization parameter: ', w_reg)
    model = Sequential()
    model.add(Conv2D(16, (5, 5), padding='valid', input_shape=(1, img_rows, img_cols), kernel_regularizer=l2(w_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (5, 5), padding='valid', kernel_regularizer=l2(w_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(w_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(w_reg)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(w_reg)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    print(model.summary())
    return model

