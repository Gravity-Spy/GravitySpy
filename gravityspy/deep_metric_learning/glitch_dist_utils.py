from keras import backend as K
import random, os, shutil
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Lambda, Activation, Reshape, Conv2D, MaxPooling2D, Flatten
from  keras.layers.core import ActivityRegularization
from keras.regularizers import l2
from functions_fusion import square_early_concatenate_feature
from sklearn.utils import shuffle


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def siamese_acc(thred):
    def inner_siamese_acc(y_true, y_pred):
        pred_res = y_pred < thred
        acc = K.mean(K.cast(K.equal(K.cast(pred_res, dtype='int32'), y_true), dtype='float32'))
        return acc
    return inner_siamese_acc


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))



def create_pairs3(data, class_indices):
    pairs = []
    labels = []
    number_of_classes = len(class_indices)
    for d in range(len(class_indices)):
        for i in range(len(class_indices[d])):

            # positive pair
            j = random.randrange(0, len(class_indices[d]))
            z1, z2 = class_indices[d][i], class_indices[d][j]
            pairs += [[data[z1], data[z2]]]

            # negative pair
            inc = random.randrange(1, number_of_classes)
            other_class_id = (d + inc) % number_of_classes
            j = random.randrange(0, len(class_indices[other_class_id])-1)
            z1, z2 = class_indices[d][i], class_indices[other_class_id][j]
            pairs += [[data[z1], data[z2]]]

            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_pairs3_gen(data, class_indices, batch_size):
    pairs1 = []
    pairs2 = []
    labels = []
    number_of_classes = len(class_indices)
    counter = 0
    while True:
        for d in range(len(class_indices)):
            for i in range(len(class_indices[d])):
                counter += 1
                # positive pair
                j = random.randrange(0, len(class_indices[d]))
                z1, z2 = class_indices[d][i], class_indices[d][j]

                pairs1.append(data[z1])
                pairs2.append(data[z2])
                labels.append(1)

                # negative pair
                inc = random.randrange(1, number_of_classes)
                other_class_id = (d + inc) % number_of_classes
                j = random.randrange(0, len(class_indices[other_class_id])-1)
                z1, z2 = class_indices[d][i], class_indices[other_class_id][j]

                pairs1.append(data[z1])
                pairs2.append(data[z2])
                labels.append(0)

                if counter == batch_size:
                    #yield np.array(pairs), np.array(labels)
                    yield [np.asarray(pairs1, np.float32), np.asarray(pairs2, np.float32)], np.asarray(labels, np.int32)
                    counter = 0
                    pairs1 = []
                    pairs2 = []
                    labels = []

def pair_gen(X, y, batch_size, negative_factor, class_indices):
    class_size = int(np.max(y) + 1)
    class_indices = [np.where(y == i)[0] for i in range(class_size)]
    number_of_classes = len(class_indices)
    start = 0
    while(1):
        if start > len(X):
            start = 0
        end = start + batch_size
        if end > len(X):
            end = len(X)
        pairs_X1 = []
        pairs_X2 = []
        pairs_y = []
        for i in range(start, end):
            Xi = X[i]
            yi = y[i]
            pairs_X1.append(Xi)
            pairs_X2.append(X[np.random.choice(class_indices[yi])])
            pairs_y.append(1)
            for ns in range(negative_factor):
                rand_class = np.random.randint(number_of_classes)
                while rand_class == yi or len(class_indices[rand_class]) == 0:
                    rand_class = np.random.randint(class_size)

                pairs_X1.append(Xi)
                pairs_X2.append(X[np.random.choice(class_indices[rand_class])])
                pairs_y.append(0)

        yield [np.asarray(pairs_X1, np.float32), np.asarray(pairs_X2, np.float32)], np.asarray(pairs_y, np.int32)
        start = start + batch_size




def create_base_network(input_dim):

    '''Base network to be shared (eq. to feature extraction).'''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.5))
    return seq


def create_base_network_conv(input_dim):
    reg_par=1e-4
    seq = Sequential()
    seq.add(Reshape((input_dim[0], input_dim[1], input_dim[2]), input_shape=(input_dim[0]*input_dim[1]*input_dim[2],)))
    #seq.add(
     #   Reshape((input_dim[0], input_dim[1], input_dim[2]), input_shape=(input_dim[0] , input_dim[1] , input_dim[2],)))

    seq.add(Conv2D(32, (5, 5), kernel_regularizer=l2(reg_par), bias_regularizer=l2(reg_par)))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D((2, 2), padding='valid'))
    seq.add(Dropout(0.5))

    seq.add(Conv2D(64, (5, 5), kernel_regularizer=l2(reg_par), bias_regularizer=l2(reg_par)))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D((2, 2), padding='valid'))
    seq.add(Dropout(0.5))

    seq.add(Flatten())
    seq.add(Dense(25, activation='relu', bias_regularizer=l2(reg_par)))
    #seq.add(Dropout(0.5))
    seq.summary()
    return seq



def compute_accuracy(predictions, labels, threshhold):
    #thred = 0.35
    return (1*((1*(predictions.ravel() < threshhold)) == labels)).mean()


def create_model_folder(file_name, save_adr):
    folder_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    print ("storing results in folder: " + folder_name)
    if not os.path.exists(save_adr + '/models/'):
        os.makedirs(save_adr + '/models/')
    full_adr = save_adr + '/models/' + folder_name
    if not os.path.exists(full_adr):
        os.mkdir(full_adr)
        print ('making full adress: '+full_adr)

    shutil.copy2(file_name, full_adr + '/')
    shutil.copy2('./GS_utils.py', full_adr + '/')
    #shutil.copy2('deep_clustering.py', full_adr + '/')
    shutil.copy2('./glitch_dist_utils.py', full_adr + '/')
    return full_adr


'''def siamese_performace_evaluation(similarity_model, pairs, pairs_labels, tag, out_file, threshhold):
    #pred_tr = similarity_model.predict([pairs[:, 0], pairs[:, 1]])
    #tr_acc = compute_accuracy(pred_tr, pairs_labels, threshhold)
    print(tag)
    per = similarity_model.evaluate([pairs[:, 0], pairs[:, 1]], pairs_labels, batch_size=32, verbose=0, sample_weight=None)
    print(per[1:])

    avg_tr_dist_pos = sum(pred_tr[pairs_labels == 1]) / len(pred_tr[pairs_labels == 1])
    avg_tr_dist_neg = sum(pred_tr[pairs_labels == 0]) / len(pred_tr[pairs_labels == 0])

    print('\n\n')
    print(tag + '\n')
    print(str(threshhold) + '\n')
    print('* Number of pairs: ', len(pairs))
    print('* Accuracy: ' + ': %0.2f%%' % (100 * tr_acc))
    print('Pos dist %0.2f%%' % avg_tr_dist_pos)
    print('Neg dist %0.2f%%' % avg_tr_dist_neg)

    out_file.write(tag + '\n')
    out_file.write(str(threshhold) + '\n')
    out_file.write('* Number of pairs: %i \n' % len(pairs))
    out_file.write('* Accuracy : %0.2f%%\n' % (100 * tr_acc))
    out_file.write(' pos dist %0.2f\n' % avg_tr_dist_pos)
    out_file.write(' neg dist %0.2f\n' % avg_tr_dist_neg)
    out_file.write('\n\n')'''


def siamese_performace_evaluation(similarity_model, pairs, pairs_labels, tag, out_file, batch_size, gen_flag, gen_func, gen_steps):
    print(tag )
    out_file.write(tag + '\n')
    if gen_flag:
        accs = similarity_model.evaluate_generator(gen_func, steps=gen_steps, verbose=1)
        #pred = similarity_model.predict_generator(gen_func, steps=gen_steps)
    else:
        accs = similarity_model.evaluate([pairs[:, 0], pairs[:, 1]], pairs_labels, batch_size=batch_size, verbose=0)
        #pred = similarity_model.predict([pairs[:, 0], pairs[:, 1]])
        print('* Number of pairs: ', len(pairs))
        out_file.write('* Number of pairs: %i \n' % len(pairs))

    #avg_tr_dist_pos = sum(pred_tr[pairs_labels == 1]) / len(pred_tr[pairs_labels == 1])
    #avg_tr_dist_neg = sum(pred_tr[pairs_labels == 0]) / len(pred_tr[pairs_labels == 0])

    #print('* Number of pairs: ', len(pairs))
    print(accs)
    #print('Pos dist %0.2f%%' % avg_tr_dist_pos)
    #print('Neg dist %0.2f%%' % avg_tr_dist_neg)
    for item in accs:
        out_file.write("%f, " % item)
    #out_file.write(' pos dist %0.2f\n' % avg_tr_dist_pos)
    #out_file.write(' neg dist %0.2f\n' % avg_tr_dist_neg)
    out_file.write('\n\n')


def reading_data(known_classes_labels, unknown_classes_labels, img_rows, img_cols, multi_view, pickle_adr ):
    from GS_utils import my_load_dataset

    if multi_view:
        dataset_name1 = 'img_5.0'
        ad1 = pickle_adr + dataset_name1 + '_class21_norm.pkl.gz'

        dataset_name2 = 'img_4.0'
        ad2 = pickle_adr + dataset_name2 + '_class21_norm.pkl.gz'

        dataset_name3 = 'img_1.0'
        ad3 = pickle_adr + dataset_name3 + '_class21_norm.pkl.gz'

        dataset_name4 = 'img_2.0'
        ad4 = pickle_adr + dataset_name4 + '_class21_norm.pkl.gz'

        datasets1 = my_load_dataset(ad1)
        test_set_x_1, test_set_y_1, test_set_name_1 = datasets1[2]
        valid_set_x_1, valid_set_y_1, valid_set_name_1 = datasets1[1]
        train_set_x_1, train_set_y_1, train_set_name_1 = datasets1[0]
        test_set_x_1 = test_set_x_1.reshape(-1, 1, img_rows, img_cols)
        train_set_x_1 = train_set_x_1.reshape(-1, 1, img_rows, img_cols)
        valid_set_x_1 = valid_set_x_1.reshape(-1, 1, img_rows, img_cols)

        datasets2 = my_load_dataset(ad2)
        test_set_x_2, test_set_y_2, test_set_name_2 = datasets2[2]
        valid_set_x_2, valid_set_y_2, valid_set_name_2 = datasets2[1]
        train_set_x_2, train_set_y_2, train_set_name_2 = datasets2[0]
        test_set_x_2 = test_set_x_2.reshape(-1, 1, img_rows, img_cols)
        train_set_x_2 = train_set_x_2.reshape(-1, 1, img_rows, img_cols)
        valid_set_x_2 = valid_set_x_2.reshape(-1, 1, img_rows, img_cols)

        datasets3 = my_load_dataset(ad3)
        test_set_x_3, test_set_y_3, test_set_name_3 = datasets3[2]
        valid_set_x_3, valid_set_y_3, valid_set_name_3 = datasets3[1]
        train_set_x_3, train_set_y_3, train_set_name_3 = datasets3[0]

        test_set_x_3 = test_set_x_3.reshape(-1, 1, img_rows, img_cols)
        train_set_x_3 = train_set_x_3.reshape(-1, 1, img_rows, img_cols)
        valid_set_x_3 = valid_set_x_3.reshape(-1, 1, img_rows, img_cols)

        datasets4 = my_load_dataset(ad4)
        test_set_x_4, test_set_y_4, test_set_name_4 = datasets4[2]
        valid_set_x_4, valid_set_y_4, valid_set_name_4 = datasets4[1]
        train_set_x_4, train_set_y_4, train_set_name_4 = datasets4[0]
        test_set_x_4 = test_set_x_4.reshape(-1, 1, img_rows, img_cols)
        train_set_x_4 = train_set_x_4.reshape(-1, 1, img_rows, img_cols)
        valid_set_x_4 = valid_set_x_4.reshape(-1, 1, img_rows, img_cols)

        concat_train = square_early_concatenate_feature(train_set_x_1, train_set_x_2, train_set_x_3, train_set_x_4,
                                                        [img_rows, img_cols])
        concat_test = square_early_concatenate_feature(test_set_x_1, test_set_x_2, test_set_x_3, test_set_x_4,
                                                       [img_rows, img_cols])
        concat_valid = square_early_concatenate_feature(valid_set_x_1, valid_set_x_2, valid_set_x_3, valid_set_x_4,
                                                        [img_rows, img_cols])
        train_set_x = concat_train
        test_set_x = concat_test
        valid_set_x = concat_valid

        train_set_y = train_set_y_1
        test_set_y = test_set_y_1
        valid_set_y = valid_set_y_1

        train_set_name = train_set_name_1
        test_set_name = test_set_name_1
        valid_set_name = valid_set_name_1

    else:
        dataset_name = 'img_1.0'
        ad = pickle_adr + dataset_name + '_class21_norm.pkl.gz'

        datasets = my_load_dataset(ad)
        test_set_x, test_set_y, test_set_name = datasets[2]
        valid_set_x, valid_set_y, valid_set_name = datasets[1]
        train_set_x, train_set_y, train_set_name = datasets[0]
        test_set_x = test_set_x.reshape(-1, 1, img_rows, img_cols)
        train_set_x = train_set_x.reshape(-1, 1, img_rows, img_cols)
        valid_set_x = valid_set_x.reshape(-1, 1, img_rows, img_cols)

    tempx = np.append(train_set_x, valid_set_x, axis=0)
    data_x = np.append(tempx, test_set_x, axis=0)
    # print(data_x.shape)

    data_x = np.reshape(data_x, (data_x.shape[0], -1))
    # print('size of train samples', data_x.shape)

    tempy = np.append(train_set_y, valid_set_y, axis=0)
    data_y = np.append(tempy, test_set_y, axis=0)

    tempn = np.append(train_set_name, valid_set_name, axis=0)
    data_name = np.append(tempn, test_set_name, axis=0)

    class_indices1 = [np.where(data_y == i)[0] for i in
                      known_classes_labels]  # from class one to number_of_known_classe for metric learning (supervised)
    class_indices2 = [np.where(data_y == i)[0] for i in
                      unknown_classes_labels]  # the rest of the samples for clustering (unsupervised)

    data_x_1 = []  # for training siamese
    data_y_1 = []
    data_name_1 = []
    for i in range(len(class_indices1)):
        for j in range(len(class_indices1[i])):
            data_x_1.append(data_x[class_indices1[i][j]])
            data_y_1.append(data_y[class_indices1[i][j]])
            data_name_1.append(data_name[class_indices1[i][j]])
    data_x_1 = np.asarray(data_x_1)
    data_y_1 = np.asarray(data_y_1)
    data_name_1 = np.asarray(data_name_1)

    data_x_2 = []  # for clustering
    data_y_2 = []
    data_name_2 = []
    for i in range(len(class_indices2)):
        for j in range(len(class_indices2[i])):
            data_x_2.append(data_x[class_indices2[i][j]])
            data_y_2.append(data_y[class_indices2[i][j]])
            data_name_2.append(data_name[class_indices2[i][j]])
    data_x_2 = np.asarray(data_x_2)
    data_y_2 = np.asarray(data_y_2)
    data_name_2 = np.asarray(data_name_2)

    data_x_1 = data_x_1.astype('float32')
    data_y_1 = data_y_1.astype('int32')
    data_x_2 = data_x_2.astype('float32')
    data_y_2 = data_y_2.astype('int32')

    data_x_1, data_y_1, data_name_1 = shuffle(data_x_1, data_y_1, data_name_1)
    data_x_2, data_y_2, data_name_2 = shuffle(data_x_2, data_y_2, data_name_2)

    # create training + test  pairs
    known_classes_indices_for_metric_learning = [np.where(data_y_1 == i)[0] for i in known_classes_labels]
    unknown_classes_indices_for_clustering = [np.where(data_y_2 == i)[0] for i in unknown_classes_labels]

    return [data_x_1, data_x_2, known_classes_indices_for_metric_learning, unknown_classes_indices_for_clustering]





