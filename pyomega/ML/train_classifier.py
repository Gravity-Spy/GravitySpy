from GS_utils import build_cnn, my_load_dataset
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from functions_fusion import square_early_concatenate_feature
import sys, getopt
import gzip, os
import cPickle
from matplotlib import use
use('agg')
from matplotlib import (pyplot as plt, cm)


'''
By Sara Bahaadini and Neda Rohani, IVPL, Northwestern University.
This function reads the  pickle files of golden_set and train a ML classifier
and write it into a model folder
explanations for options are given in the code
'''


def main(batch_size,nb_epoch,train_flag,pickle_adr,save_address,number_of_classes,verbose):

    np.random.seed(1986)  # for reproducibility

    img_rows, img_cols = 47, 57
    nb_classes = number_of_classes

    if verbose:
        print 'save adress', save_address
        print 'test flag', test_flag
    if not os.path.exists(save_address):
        if verbose:
            print ('making... ' + save_address)
        os.makedirs(save_address)

    dataset_name1 = 'img_5.0'
    ad1 = pickle_adr + dataset_name1 + '_class' + str(nb_classes) + '_norm.pkl.gz'

    dataset_name2 = 'img_4.0'
    ad2 = pickle_adr + dataset_name2 + '_class' + str(nb_classes) + '_norm.pkl.gz'

    dataset_name3 = 'img_1.0'
    ad3 = pickle_adr + dataset_name3 + '_class' + str(nb_classes) + '_norm.pkl.gz'

    dataset_name4 = 'img_2.0'
    ad4 = pickle_adr + dataset_name4 + '_class' + str(nb_classes) + '_norm.pkl.gz'

    print 'batch size', batch_size
    print 'iteration', nb_epoch
    print 'flag mode', train_flag
    print 'Reading the pickles from: ', pickle_adr
    print 'pickle_1: ', ad1
    print 'pickle_2: ', ad2
    print 'pickle_3: ', ad3
    print 'pickle_4: ', ad4
    print 'saving the trained model in: ', save_address

   # if not os.path.exists(save_address):
    #    print ('making... ' + save_address)
    #    os.makedirs(save_address)

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

    assert (test_set_y_2 == test_set_y_1).all()
    assert (valid_set_y_2 == valid_set_y_1).all()
    assert (train_set_y_2 == train_set_y_1).all()

    assert (test_set_y_2 == test_set_y_3).all()
    assert (valid_set_y_2 == valid_set_y_3).all()
    assert (train_set_y_2 == train_set_y_3).all()

    assert (test_set_y_3 == test_set_y_4).all()
    assert (valid_set_y_3 == valid_set_y_4).all()
    assert (train_set_y_3 == train_set_y_4).all()

    assert (test_set_name_1 == test_set_name_2).all()
    assert (test_set_name_2 == test_set_name_3).all()
    assert (test_set_name_3 == test_set_name_4).all()

    assert (valid_set_name_1 == valid_set_name_2).all()
    assert (valid_set_name_2 == valid_set_name_3).all()
    assert (valid_set_name_3 == valid_set_name_4).all()

    assert (train_set_name_1 == train_set_name_2).all()
    assert (train_set_name_2 == train_set_name_3).all()
    assert (train_set_name_3 == train_set_name_4).all()

    cat_train_set_y_1 = np_utils.to_categorical(train_set_y_1, nb_classes)
    cat_valid_set_y_1 = np_utils.to_categorical(valid_set_y_1, nb_classes)
    cat_test_set_y_1 = np_utils.to_categorical(test_set_y_1, nb_classes)



    concat_train = square_early_concatenate_feature(train_set_x_1, train_set_x_2,train_set_x_3, train_set_x_4, [img_rows, img_cols])
    concat_test = square_early_concatenate_feature(test_set_x_1, test_set_x_2, test_set_x_3, test_set_x_4,[img_rows, img_cols])
    concat_valid = square_early_concatenate_feature(valid_set_x_1, valid_set_x_2, valid_set_x_3, valid_set_x_4,[img_rows, img_cols])


    cnn1 = build_cnn(img_rows*2, img_cols*2)
    final_model = Sequential()
    final_model.add(cnn1)
    final_model.add(Dense(nb_classes, activation='softmax'))

    #model_optimizer = RMSprop(lr=0.1)
    final_model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  # optimizer=model_optimizer,
                  metrics=['accuracy'])

    model_adr = save_address + '/models/'
    if not os.path.exists(model_adr):
        print ('making models address ... ' + model_adr)
        os.makedirs(model_adr)

    acc_checker = ModelCheckpoint(model_adr + "/best_weights.h5", monitor='val_acc', verbose=1,
                                  save_best_only=True, mode='max', save_weights_only=True)

    if train_flag:
        print(concat_train.shape[0], 'train samples')
        print(concat_valid.shape[0], 'validation samples')
        print(concat_test.shape[0], 'test samples')

        final_model.fit(concat_train, cat_train_set_y_1,
                            batch_size=batch_size, epochs=nb_epoch, verbose=1,
                            validation_data=(concat_valid, cat_valid_set_y_1), callbacks=[acc_checker])

        final_model.load_weights(model_adr + "/best_weights.h5")
        score = final_model.evaluate(concat_test, cat_test_set_y_1, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        #print final_model.summary()
        print 'done'
    else:
        all_data_for_train = np.append(concat_train, concat_valid, axis=0)
        all_data_for_train = np.append(all_data_for_train, concat_test, axis=0)
        all_label_for_train = np.append(cat_train_set_y_1, cat_valid_set_y_1, axis=0)
        all_label_for_train = np.append(all_label_for_train, cat_test_set_y_1, axis=0)

        print('Number of training samples:', all_data_for_train.shape[0])
        final_model.fit(all_data_for_train, all_label_for_train,
                    batch_size=batch_size, epochs=nb_epoch, verbose=1,
                    validation_data=(concat_valid, cat_valid_set_y_1), callbacks=[acc_checker])
        final_model.load_weights(model_adr + "/best_weights.h5")



    # save model and weights
    json_string = final_model.to_json()
    f = gzip.open(save_address + '/model.pklz', 'wb')
    cPickle.dump(json_string, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    final_model.save_weights(save_address + '/model_weights.h5', overwrite=True)


if __name__ == "__main__":
   print 'Start ...'
   main(batch_size,nb_epoch,train_flag,pickle_adr,save_address,number_of_classes,verbose)
   print 'Done!'

