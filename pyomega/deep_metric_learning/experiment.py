from GS_utils import my_load_dataset
from sklearn.cluster import KMeans
import sys, getopt, os
import numpy as np
from sklearn.utils import shuffle
from keras.layers import Convolution2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda, Activation, Reshape
from keras.optimizers import RMSprop
from glitch_dist_utils import create_pairs3, compute_accuracy, contrastive_loss, eucl_dist_output_shape, siamese_acc, create_pairs3_gen
from glitch_dist_utils import euclidean_distance, create_base_network, create_base_network_conv, create_model_folder, siamese_performace_evaluation
from functions_fusion import square_early_concatenate_feature
from datetime import datetime

def main(argv):
    print(datetime.now().strftime("%Y%m%d-%H%M%S"))
    print('training using all classes for Gravity Spy framework')
    # np.random.seed(1337)  # for reproducibility
    #known_classes_labels = [0, 1, 2, 4, 5, 6, 7, 9, 13, 14]
    #unknown_classes_labels = [15, 16, 17, 18, 19]
    #known_classes_labels = [0, 1, 2, 4, 5, 6, 7, 9, 13, 14,15, 16, 17, 18, 19, 20]
    #unknown_classes_labels = [19, 20]
    known_classes_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 19, 20]
    known_classes_labels = list(range(0, 21))
    unknown_classes_labels = [18, 9, 16]

    img_rows, img_cols = 140, 170

    nb_epoch = 150
    threshhold = 0.4
    size_of_batch = 30
    model_check_point = 0
    multi_view = 1
    gen_for_train = 1
    gen_for_test = 1

    #pickle_adr = './pickle/april10_pickles/img_1.0_class21_norm.pkl.gz'
    pickle_adr = '/home/sara/ciera_phase2/pickles_for_similarity/pickle_2017-08-07_res0.3/'
    save_adr = './model_for_clustering_alg_aug7/'
    if not os.path.exists(save_adr):
        print('making... ' + save_adr)
        os.makedirs(save_adr)
    print('multi_view =', multi_view)
    print('model check point = ', model_check_point)
    if multi_view:
        dataset_name1 = 'img_5.0'
        ad1 = pickle_adr + dataset_name1 + '_class21_norm.pkl.gz'

        dataset_name2 = 'img_4.0'
        ad2 = pickle_adr + dataset_name2 + '_class21_norm.pkl.gz'

        dataset_name3 = 'img_1.0'
        ad3 = pickle_adr + dataset_name3 + '_class21_norm.pkl.gz'

        dataset_name4 = 'img_2.0'
        ad4 = pickle_adr + dataset_name4 + '_class21_norm.pkl.gz'

        print('batch size', size_of_batch)
        print('iteration', nb_epoch)
        print('generator for train', gen_for_train)
        print('generator for test', gen_for_test)
        print('Reading the pickles from: ', pickle_adr)
        print('pickle_1: ', ad1)
        print('pickle_2: ', ad2)
        print('pickle_3: ', ad3)
        print('pickle_4: ', ad4)
        print('saving the trained model in: ', save_adr)

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

#        print(train_set_x.shape)

    tempx = np.append(train_set_x, valid_set_x, axis=0)
    data_x = np.append(tempx, test_set_x, axis=0)
    #print(data_x.shape)

    data_x = np.reshape(data_x, (data_x.shape[0], -1))
    #print('size of train samples', data_x.shape)

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


    if gen_for_train:
        tr_pairs, tr_y = create_pairs3(data_x_1, known_classes_indices_for_metric_learning)
    if gen_for_test:
        te_pairs2, te_y2 = create_pairs3(data_x_1, known_classes_indices_for_metric_learning)
        te_pairs, te_y = create_pairs3(data_x_2, unknown_classes_indices_for_clustering)

    # making the siamese model
    if multi_view:
        img_rows = 2 * img_rows
        img_cols = 2 * img_cols

    base_network = create_base_network_conv([1, img_rows, img_cols])
    input_dim = img_cols * img_rows
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    similarity_model = Model(inputs=[input_a, input_b], outputs=distance)
    semantic_idx_model = Model(inputs=[input_a], outputs=processed_a)

    # train the siamese model
    rms = RMSprop()
    similarity_model.compile(loss=contrastive_loss, optimizer=rms, metrics=[siamese_acc(0.4), siamese_acc(0.5), siamese_acc(0.6), siamese_acc(0.7)])

    full_model_adr = create_model_folder(__file__, save_adr)
    # acc_checker = ModelCheckpoint(full_model_adr + "/best_weights.h5", monitor='val_loss',
    #                             verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    loss_checker = ModelCheckpoint(full_model_adr + "/best_weights_loss.h5", monitor='val_loss',
                                   verbose=1, save_best_only=True, mode='auto', save_weights_only=True)

    acc_checker = ModelCheckpoint(full_model_adr + "/best_weights_acc.h5", monitor='inner_siamese_acc_2',
                                  verbose=1, save_best_only=True, mode='auto', save_weights_only=True)
    callbacks = [acc_checker, loss_checker]

    if gen_for_train:
        train_negative_factor = 1
        test_negative_factor = 1
        train_generator = create_pairs3_gen(data_x_1, known_classes_indices_for_metric_learning, size_of_batch)
        valid_generator = create_pairs3_gen(data_x_2, unknown_classes_indices_for_clustering, size_of_batch)
        # the samples from data_x_2 should be separated for test and valid in future

        train_batch_num = (len(data_x_1) * (train_negative_factor + 1)) / size_of_batch
        print('train batch num', train_batch_num)
        valid_batch_num = (len(data_x_2) * (test_negative_factor + 1)) / size_of_batch

        similarity_model.fit_generator(train_generator, validation_data=valid_generator, verbose=2,
                            steps_per_epoch=train_batch_num, validation_steps=valid_batch_num, epochs=nb_epoch)
    else:
        similarity_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                         validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
                         batch_size=size_of_batch, epochs=nb_epoch, verbose=2, callbacks=callbacks)

    # without model check point
    final_model_adr = full_model_adr + "/without_check_point"
    if not os.path.exists(final_model_adr):
        print('making... ' + final_model_adr)
        os.makedirs(final_model_adr)
    similarity_model.save(final_model_adr + '/similarity_metric_model.h5')
    semantic_idx_model.save(final_model_adr + '/semantic_idx_model.h5')
    out_file = open(final_model_adr + '/out.txt', "w")

    known_classes_negative_factor = 1
    unknown_classes_negative_factor = 1
    known_classes_generator = create_pairs3_gen(data_x_1, known_classes_indices_for_metric_learning, size_of_batch)
    unknown_classes_generator = create_pairs3_gen(data_x_2, unknown_classes_indices_for_clustering, size_of_batch)

    known_classes_batch_num = (len(data_x_1) * (known_classes_negative_factor + 1)) / size_of_batch
    unknown_classes_batch_num = (len(data_x_2) * (unknown_classes_negative_factor + 1)) / size_of_batch

    # performance on training pairs
    if not gen_for_test:
        siamese_performace_evaluation(similarity_model, tr_pairs, tr_y, 'Without check point, training pairs ',
                                  out_file, size_of_batch, 0, 0, 0)

    # performance on test pairs from known classes
    siamese_performace_evaluation(similarity_model, te_pairs2, te_y2, 'Without check point, test pairs (known classes)',
                                  out_file, size_of_batch, gen_for_test, known_classes_generator, known_classes_batch_num)

    # performance on test pairs from unknown classes
    siamese_performace_evaluation(similarity_model, te_pairs, te_y, 'Without check point, test pairs (unknown classes)', out_file,
                                  size_of_batch, gen_for_test, unknown_classes_generator, unknown_classes_batch_num)

    if model_check_point:
        # evaluating the model based on the best accuracy
        similarity_model.load_weights(full_model_adr + "/best_weights_acc.h5")
        final_model_adr = full_model_adr + "/final_model_acc"
        if not os.path.exists(final_model_adr):
            print('making... ' + final_model_adr)
            os.makedirs(final_model_adr)
        similarity_model.save(final_model_adr + '/similarity_metric_model.h5')
        semantic_idx_model.save(final_model_adr + '/semantic_idx_model.h5')
        out_file = open(final_model_adr + '/out_best_acc.txt', "w")

        # performance on training pairs
        siamese_performace_evaluation(similarity_model, tr_pairs, tr_y, 'Best model on acc, training pairs ', out_file, threshhold)

        # performance on test pairs from unknown classes
        siamese_performace_evaluation(similarity_model, te_pairs, te_y, 'Best model on acc, test pairs (unknown classes)', out_file, threshhold)
        siamese_performace_evaluation(similarity_model, te_pairs, te_y, '**', out_file, 0.5)
        siamese_performace_evaluation(similarity_model, te_pairs, te_y, '**', out_file, 0.6)
        siamese_performace_evaluation(similarity_model, te_pairs, te_y, '**', out_file, 0.7)

        # performance on test pairs from unknown classes
        siamese_performace_evaluation(similarity_model, te_pairs2, te_y2, 'Best model on acc, test pairs (known classes)', out_file, threshhold)
        out_file.close()

        # evaluating the model based on the best accuracy
        similarity_model.load_weights(full_model_adr + "/best_weights_loss.h5")
        final_model_adr = full_model_adr + "/final_model_loss"
        if not os.path.exists(final_model_adr):
            print('making... ' + final_model_adr)
            os.makedirs(final_model_adr)
        similarity_model.save(final_model_adr + '/similarity_metric_model.h5')
        semantic_idx_model.save(final_model_adr + '/semantic_idx_model.h5')
        out_file = open(final_model_adr + '/out_best_loss.txt', "w")

        # performance on training pairs
        siamese_performace_evaluation(similarity_model, tr_pairs, tr_y, 'Best model on loss, training pairs ', out_file, threshhold)

        # performance on test pairs from unknown classes
        siamese_performace_evaluation(similarity_model, te_pairs2, te_y2, 'Best model on loss, test pairs (known classes)', out_file, threshhold)

        # performance on test pairs from unknown classes
        siamese_performace_evaluation(similarity_model, te_pairs, te_y, 'Best model on loss, test pairs (unknown classes)', out_file, threshhold)
        siamese_performace_evaluation(similarity_model, te_pairs, te_y, '**', out_file, 0.5)
        siamese_performace_evaluation(similarity_model, te_pairs, te_y, '**', out_file, 0.6)
        siamese_performace_evaluation(similarity_model, te_pairs, te_y, '**', out_file, 0.7)



    ####################################### best val ######################################
    '''similarity_model.load_weights(full_model_adr + "/best_weights_loss.h5")
    pred_tr = similarity_model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred_tr, tr_y, threshhold)

    final_model_adr = full_model_adr + "/final_model_loss"
    if not os.path.exists(final_model_adr):
        print('making... ' + final_model_adr)
        os.makedirs(final_model_adr)

    similarity_model.save(final_model_adr + '/similarity_metric_model.h5')
    semantic_idx_model.save(final_model_adr + '/semantic_idx_model.h5')
    out_file = open(final_model_adr + '/out_best_loss.txt', "w")

    avg_tr_dist_pos = sum(pred_tr[tr_y == 1]) / len(pred_tr[tr_y == 1])
    avg_tr_dist_neg = sum(pred_tr[tr_y == 0]) / len(pred_tr[tr_y == 0])

    print('*********************************************************')
    print('* Number of training pairs: ', len(tr_pairs))
    print('* Accuracy on training set (on best loss): %0.2f%%' % (100 * tr_acc))
    out_file.write('* Accuracy on training set (on best loss): %0.2f%%' % (100 * tr_acc))
    out_file.write('train pos dist %0.2f%% (on best loss)' % avg_tr_dist_pos)
    out_file.write('train neg dist %0.2f%% (on best loss)' % avg_tr_dist_neg)
    print('train pos dist %0.2f%% (on best loss)' % avg_tr_dist_pos)
    print('train neg dist %0.2f%% (on best loss)' % avg_tr_dist_neg)

    similarity_model.load_weights(full_model_adr + "/best_weights.h5")
    final_model_adr = full_model_adr + "/final_model"
    if not os.path.exists(final_model_adr):
        print('making... ' + final_model_adr)
        os.makedirs(final_model_adr)
    out_file = open(final_model_adr + '/out.txt', "w")

    similarity_model.save(final_model_adr + '/similarity_metric_model_for_clustering.h5')
    semantic_idx_model.save(final_model_adr + '/semantic_idx_model_for_clustering.h5')

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
    out_file.write('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('train pos dist', avg_tr_dist_pos)
    print('train neg dist', avg_tr_dist_neg)
    print('*****************************************')
    # print ('* Number of testing pairs: ',len(te_pairs))
    print('* Accuracy on test set (from known classes): %0.2f%%' % (100 * te_acc2))
    out_file.write('* Accuracy on training set: %0.2f%%' % (100 * te_acc2))
    print('test pos dist', avg_te2_dist_pos)
    print('test neg dist', avg_te2_dist_neg)
    print('*****************************************')
    print('* Accuracy on test set (from unknown classes): %0.2f%%' % (100 * te_acc))
    out_file.write('* Accuracy on training set: %0.2f%%' % (100 * te_acc))
    print('test pos dist', avg_te_dist_pos)
    print('test neg dist', avg_te_dist_neg)'''


if __name__ == "__main__":
    main(sys.argv[1:])
 