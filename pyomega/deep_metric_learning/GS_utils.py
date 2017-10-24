

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rescale
import os
import numpy as np

import time
import sys
from keras.regularizers import l2
import numpy
import gzip
import pickle
import keras.callbacks

from keras.layers import Conv2D, MaxPooling2D, Flatten


from sklearn.cross_validation import StratifiedShuffleSplit
from functools import reduce
from sklearn.utils import shuffle

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Activation, Reshape
from keras.optimizers import RMSprop
from glitch_dist_utils import create_pairs3, compute_accuracy, contrastive_loss, eucl_dist_output_shape
from glitch_dist_utils import euclidean_distance, create_base_network, create_base_network_conv, create_model_folder


def build_cnn(img_rows, img_cols):
    W_reg = 2*(1e-4)
    print ('regularization parameter: ', W_reg)

    model = Sequential()
    model.add(Conv2D(16, (5, 5), padding='valid', input_shape=(1, img_rows, img_cols), kernel_regularizer=l2(W_reg)))
    #model.add(BatchNormalization())
    # model.add(Conv2D(128, (5, 5), activation='relu', padding='valid'))
    #model.add(BatchNormalization(axis=axis_b))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    '''model.add(Conv2D(64, (5, 5), activation='relu', padding='valid', kernel_regularizer=l2(W_reg)))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))'''

    model.add(Conv2D(32, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    #model.add(BatchNormalization())
    #model.add(BatchNormalization(axis=axis_b))
    model.add(Activation("relu"))
    #model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    # model.add(BatchNormalization())
    # model.add(BatchNormalization(axis=axis_b))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    # model.add(BatchNormalization())
    # model.add(BatchNormalization(axis=axis_b))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(W_reg)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    print (model.summary())
    return model

def main(argv):
    print ('gravityspy_utils')

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

def load_dataset_unlabelled_glitches(dataset):
    with gzip.open(dataset, 'rb') as f:
        try:
            test_set = pickle.load(f, encoding='latin1')
        except:
            test_set = pickle.load(f)
    print('The size of test set is: ', len(test_set))
    return test_set


def my_save_dataset_test(save_adress, imgs, labels, names, str_name):

    labels = numpy.array(labels)
    names = numpy.array(names)

    input_te = imgs
    target_te = labels
    names_te = names

    div_norm = 255.0
    out = [numpy.divide(input_te, div_norm), target_te, names_te]

    f = gzip.open(save_adress + 'img' +str_name + 'class' + str(max(labels)+1) + '_norm.pkl.gz', 'wb')
    pickle.dump(out, f, protocol=2)
    f.close()


def my_save_dataset(save_adress, new_ind, imgs, labels, names, str_name ):
    data_size = imgs.shape[0]
    "Partitioning into train, validation and test sets"
    trainNumber = data_size * 0.75
    validNumber = data_size * 0.125
    testNumber = data_size * 0.125

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
	
def my_save_dataset_stratified(save_adress, imgs, labels, names, str_name):

    my_file2 = open(save_adress + 'dataset_info.txt', 'w')
    skf = StratifiedShuffleSplit(labels, 1, test_size=0.3, random_state=0)
    for train_index, test_index in skf:
        trainset = train_index

    labels = numpy.array(labels)
    names = numpy.array(names)

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

    input_val = sub_imgs[valisdet]
    target_val = sub_labels[valisdet]
    names_val = sub_names[valisdet]
    my_file2.write('\n\n' + 'Validation set: \n')
    for name in names_val:
        my_file2.write(str(name) + ',' + '\n')

    input_te = sub_imgs[testset]
    target_te = sub_labels[testset]
    names_te = sub_names[testset]
    my_file2.write('\n\n' + 'Test set: \n' + '\n')
    for name in names_te:
        my_file2.write(str(name) + ',' + '\n')

    my_file2.close()
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



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append([logs.get('loss'), logs.get('acc'), logs.get('val_loss'), logs.get('val_acc')])

'''def my_read_image(data_path, dataset_str):

    classes = os.listdir(data_path)
    classes = [c for c in classes if not c.startswith('.') ]
    imgs = []
    labels = []
    names = []

    #np.savetxt('/Users/Sara/Desktop/pickle/pkl/all25/path_folders_'+ dataset_str +'.csv', classes, delimiter=',', fmt='%s')
    for index, item in enumerate(classes):
      path = data_path + classes[index]
      #if os.path.isdir(path):
      if 1:
        print(path)
        files = os.listdir(path)
        files = [f for f in files if not f.startswith('.')]
        #np.savetxt('/Users/Sara/Desktop/pickle/pkl/all/path_files_' + dataset_str + '.csv', files, delimiter=',', fmt='%s')
        for f in files:
            if dataset_str in f:
                test = io.imread(os.path.join(path, f))
                test = test[66:532, 105:671, :]
                #plt.imshow(test)
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
    return [imgs, labels, names]'''


def my_read_image2(save_address, data_path, dataset_str, resolution, for_similarity_measure_flag):

    classes = os.listdir(data_path)
    classes = [c for c in classes if not c.startswith('.')]
    if for_similarity_measure_flag:
        classes = [c for c in classes if not 'None_of_the_Above' in c]
    imgs = []
    labels = []
    names = []
    my_file = open(save_address + 'classes_distribution.txt', 'w')
    # np.savetxt('/Users/Sara/Desktop/pickle/pkl/all25/path_folders_'+ dataset_str +'.csv', classes, delimiter=',', fmt='%s')
    for index, item in enumerate(classes):
        counter = 0
        path = data_path + classes[index]
        # if os.path.isdir(path):
        if 1:
            print(path)
            files = os.listdir(path)
            files = [f for f in files if not f.startswith('.')]
            # np.savetxt('/Users/Sara/Desktop/pickle/pkl/all/path_files_' + dataset_str + '.csv', files, delimiter=',', fmt='%s')
            for f in files:
                if dataset_str in f:
                    test = io.imread(os.path.join(path, f))
                    test = test[66:532, 105:671, :]
                   # plt.imshow(test)
                    test = rgb2gray(test)
                    test = rescale(test, [resolution, resolution], mode='constant')
                    # test = resize(test, (50, 50))
                    # num_feature = test.shape
                    dim = numpy.int(reduce(lambda x, y: x * y, test.shape))
                    imgs.append(numpy.reshape(test, (dim)))  # vectorizing
                    names.append(f[0:13])  # its name
                    labels.append(index)
                    counter = counter + 1

        my_file.write(classes[index] + ': ' + str(counter)+'\n')

    my_file.close()
   # np.savetxt('classes_distribution_' + dataset_str + '.csv', classes[index] + ': ' + str(counter),fmt='%s')
    np.savetxt(save_address + 'class2idx_'+ dataset_str+'.csv', classes, delimiter=',', fmt='%s')
    return [imgs, labels, names, classes]




'''def my_read_image2(data_path, dataset_str):

    classes = os.listdir(data_path)
    classes = [c for c in classes if not c.startswith('.')]

    imgs = []
    labels = []
    names = []

    # np.savetxt('/Users/Sara/Desktop/pickle/pkl/all25/path_folders_'+ dataset_str +'.csv', classes, delimiter=',', fmt='%s')
    for index, item in enumerate(classes):
        path = data_path + classes[index]
        # if os.path.isdir(path):
        if 1:
            print(path)
            files = os.listdir(path)
            files = [f for f in files if not f.startswith('.')]
            # np.savetxt('/Users/Sara/Desktop/pickle/pkl/all/path_files_' + dataset_str + '.csv', files, delimiter=',', fmt='%s')
            for f in files:
                if dataset_str in f:
                    test = io.imread(os.path.join(path, f))
                    test = test[66:532, 105:671, :]
                    # plt.imshow(test)
                    test = rgb2gray(test)
                    test = rescale(test, [0.1, 0.1])
                    # test = resize(test, (50, 50))
                    # num_feature = test.shape
                    dim = numpy.int(reduce(lambda x, y: x * y, test.shape))
                    imgs.append(numpy.reshape(test, (dim)))  # vectorizing
                    names.append(f[0:13])  # its name
                    labels.append(index)
                    # labels.append(1)
    np.savetxt('class2idx.csv', classes, delimiter=',', fmt='%s')
    return [imgs, labels, names, classes] '''


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





def build_partial_cnn1(img_rows, img_cols):
    model = Sequential()
    #model.add(Convolution2D(nb_filter=100, nb_row=5, nb_col=5,
    model.add(Conv2D(10, (2, 2), activation='linear',
                            padding='valid',
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
    model.add(Conv2D(100, (5, 5),
                            activation='linear',
                            padding='valid',
                            input_shape=(n, img_rows, img_cols)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(100, (5, 5), activation='linear',
                            padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    return model
	
	
def model_making_for_clustering(known_classes_labels, unknown_classes_labels, pickle_adr, save_adr):
    nb_epoch = 200
    threshhold = 0.4
    size_of_batch = 128
    img_rows, img_cols = 47, 57
    input_dim = 2679
    if not os.path.exists(save_adr):
        print ('making... ' + save_adr)
        os.makedirs(save_adr)
    datasets = my_load_dataset(pickle_adr)
    test_set_x, test_set_y, test_set_name = datasets[2]
    valid_set_x, valid_set_y, valid_set_name = datasets[1]
    train_set_x, train_set_y, train_set_name = datasets[0]

    tempx = np.append(train_set_x, valid_set_x, axis=0)
    data_x = np.append(tempx, test_set_x, axis=0)

    tempy = np.append(train_set_y, valid_set_y, axis=0)
    data_y = np.append(tempy, test_set_y, axis=0)

    tempn = np.append(train_set_name, valid_set_name, axis=0)
    data_name = np.append(tempn, test_set_name, axis=0)

    class_indices_1 = [np.where(data_y == i)[0] for i in
                       known_classes_labels]  # from class one to number_of_known_classe for metric learning (supervised)
    class_indices_2 = [np.where(data_y == i)[0] for i in
                       unknown_classes_labels]  # the rest of the samples for clustering (unsupervised)

    data_x_1 = []  # for metric learning
    data_y_1 = []
    data_name_1 = []
    for i in range(len(class_indices_1)):
        for j in range(len(class_indices_1[i])):
            data_x_1.append(data_x[class_indices_1[i][j]])
            data_y_1.append(data_y[class_indices_1[i][j]])
            data_name_1.append(data_name[class_indices_1[i][j]])
    data_x_1 = np.asarray(data_x_1)
    data_y_1 = np.asarray(data_y_1)
    data_name_1 = np.asarray(data_name_1)

    data_x_2 = []  # for clustering
    data_y_2 = []
    data_name_2 = []
    for i in range(len(class_indices_2)):
        for j in range(len(class_indices_2[i])):
            data_x_2.append(data_x[class_indices_2[i][j]])
            data_y_2.append(data_y[class_indices_2[i][j]])
            data_name_2.append(data_name[class_indices_2[i][j]])
    data_x_2 = np.asarray(data_x_2)
    data_y_2 = np.asarray(data_y_2)
    data_name_2 = np.asarray(data_name_2)

    data_x_1 = data_x_1.astype('float32')
    data_x_2 = data_x_2.astype('float32')

    data_x_1, data_y_1, data_name_1 = shuffle(data_x_1, data_y_1, data_name_1)
    data_x_2, data_y_2, data_name_2 = shuffle(data_x_2, data_y_2, data_name_2)

    # create training + test positive and negative pairs
    known_classes_indices_for_metric_learning = [np.where(data_y_1 == i)[0] for i in known_classes_labels]
    # tr_pairs, tr_y = create_pairs2(data_x_1, known_classes_indices_for_metric_learning,
    #                               1, number_of_known_classes, number_of_unknown_classes)
    tr_pairs, tr_y = create_pairs3(data_x_1, known_classes_indices_for_metric_learning)
    # te_pairs2, te_y2 = create_pairs2(data_x_1, known_classes_indices_for_metric_learning,
    #                                 1, number_of_known_classes, number_of_unknown_classes)
    te_pairs2, te_y2 = create_pairs3(data_x_1, known_classes_indices_for_metric_learning)
    unknown_classes_indices_for_clustering = [np.where(data_y_2 == i)[0] for i in unknown_classes_labels]
    # te_pairs, te_y = create_pairs2(data_x_2, unknown_classes_indices_for_clustering, 0,
    #                               number_of_known_classes, number_of_unknown_classes)
    te_pairs, te_y = create_pairs3(data_x_2, unknown_classes_indices_for_clustering)

    # network definition
    base_network = create_base_network_conv([1, img_rows, img_cols])

    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    similarity_model = Model(input=[input_a, input_b], output=distance)
    semantic_idx_model = Model(input=[input_a], output=processed_a)

    # train
    rms = RMSprop()
    similarity_model.compile(loss=contrastive_loss, optimizer=rms)

    full_model_adr = create_model_folder(save_adr)
    acc_checker = ModelCheckpoint(full_model_adr + "/best_weights.h5", monitor='val_loss',
                                  verbose=1, save_best_only=True, mode='min', save_weights_only=True)

    similarity_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                         validation_data=([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y),
                         batch_size=size_of_batch, nb_epoch=nb_epoch, verbose=1, callbacks=[acc_checker])

    # compute final accuracy on training and test sets
    similarity_model.load_weights(full_model_adr + "/best_weights.h5")
    final_model_adr = full_model_adr + "/final_model"
    if not os.path.exists(final_model_adr):
        print ('making... ' + final_model_adr)
        os.makedirs(final_model_adr)
    out_file = open(final_model_adr + '/out.txt', "w")

    similarity_model.save(final_model_adr + '/similarity_metric_model_for_clustering.h5')
    semantic_idx_model.save(final_model_adr + '/semantic_idx_model_for_clustering.h5')

    out = [data_x_1, data_y_1, data_name_1, data_x_2, data_y_2, data_name_2]
    f = gzip.open(final_model_adr + '/data1_data2.pkl', 'wp')
    pickle.dump(out, f, protocol=2)
    f.close()

    pred_tr = similarity_model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred_tr, tr_y, threshhold)

    # plt.plot(pred_tr[tr_y==1], np.zeros_like(pred_tr[tr_y==1])+0.,'r*')
    # plt.plot(pred_tr[tr_y==0], np.zeros_like(pred_tr[tr_y==0])+0.,'g*')
    # plt.plot(pred_tr[pred_tr < 0.5], np.zeros_like(pred_tr[pred_tr < 0.5])+0.,'r*')
    # plt.plot(pred_tr[pred_tr > 0.5], np.zeros_like(pred_tr[pred_tr > 0.5])+0.,'bo')
    # plt.show()

    pred_te2 = similarity_model.predict([te_pairs2[:, 0], te_pairs2[:, 1]])
    te_acc2 = compute_accuracy(pred_te2, te_y2, threshhold)

    pred_te = similarity_model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred_te, te_y, threshhold)

    avg_tr_dist_pos = sum(pred_tr[tr_y == 1]) / len(pred_tr[tr_y == 1])
    avg_tr_dist_neg = sum(pred_tr[tr_y == 0]) / len(pred_tr[tr_y == 0])

    avg_te_dist_pos = sum(pred_te[te_y == 1]) / len(pred_te[te_y == 1])
    avg_te_dist_neg = sum(pred_te[te_y == 0]) / len(pred_te[te_y == 0])

    avg_te2_dist_pos = sum(pred_te2[te_y2 == 1]) / len(pred_te2[te_y2 == 1])
    avg_te2_dist_neg = sum(pred_te2[te_y2 == 0]) / len(pred_te2[te_y2 == 0])

    print('*****************************************')
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    out_file.write('* Accuracy on training set: %0.2f%% \n' % (100 * tr_acc))
    print ('train pos dist', avg_tr_dist_pos)
    print ('train neg dist', avg_tr_dist_neg)
    print('*****************************************')
    # print ('* Number of testing pairs: ',len(te_pairs))
    print('* Accuracy on test set (from known classes): %0.2f%%' % (100 * te_acc2))
    out_file.write('* Accuracy on training set: %0.2f%% \n' % (100 * te_acc2))
    print ('test pos dist', avg_te2_dist_pos)
    print ('test neg dist', avg_te2_dist_neg)
    print('*****************************************')
    print('* Accuracy on test set (from unknown classes): %0.2f%%' % (100 * te_acc))
    out_file.write('* Accuracy on training set: %0.2f%% \n' % (100 * te_acc))
    print ('test pos dist', avg_te_dist_pos)
    print ('test neg dist', avg_te_dist_neg)

    return final_model_adr, data_x_2, data_y_2, data_name_2, out_file

if __name__ == "__main__":
   main(sys.argv[1:])
