from GS_utils import build_cnn, my_load_dataset, create_model_folder, concatenate_views
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
#from functions_fusion import square_early_concatenate_feature
import sys, getopt
import gzip, os

'''
By Sara Bahaadini
This function reads the  pickle files of golden_set and train a ML classifier
and write it into a model folder
'''


def main(batch_size, nb_epoch, all_data_for_train_flag, pickle_adr, save_address,
         number_of_classes, image_size, best_model_based_validset, verbose):

    np.random.seed(1986)  # for reproducibility

    img_rows, img_cols = image_size[0], image_size[1]
    nb_classes = number_of_classes

    if not os.path.exists(save_address):
        if verbose:
            print('making... ' + save_address)
        os.makedirs(save_address)

    dataset_name1 = 'img_5.0'
    ad1 = pickle_adr + dataset_name1 + '_class' + str(nb_classes) + '_norm.pkl.gz'

    dataset_name2 = 'img_4.0'
    ad2 = pickle_adr + dataset_name2 + '_class' + str(nb_classes) + '_norm.pkl.gz'

    dataset_name3 = 'img_1.0'
    ad3 = pickle_adr + dataset_name3 + '_class' + str(nb_classes) + '_norm.pkl.gz'

    dataset_name4 = 'img_2.0'
    ad4 = pickle_adr + dataset_name4 + '_class' + str(nb_classes) + '_norm.pkl.gz'

    if verbose:
        print('batch size', batch_size)
        print('iteration', nb_epoch)
        print('flag mode', all_data_for_train_flag)
        print('Reading the pickles from: ', pickle_adr)
        print('pickle_1: ', ad1)
        print('pickle_2: ', ad2)
        print('pickle_3: ', ad3)
        print('pickle_4: ', ad4)
        print('saving the trained model in: ', save_address)

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

    cat_train_set_y_1 = np_utils.to_categorical(train_set_y_1, nb_classes)
    cat_valid_set_y_1 = np_utils.to_categorical(valid_set_y_1, nb_classes)
    cat_test_set_y_1 = np_utils.to_categorical(test_set_y_1, nb_classes)

    concat_train = concatenate_views(train_set_x_1, train_set_x_2,train_set_x_3, train_set_x_4, [img_rows, img_cols])
    concat_test = concatenate_views(test_set_x_1, test_set_x_2, test_set_x_3, test_set_x_4,[img_rows, img_cols])
    concat_valid = concatenate_views(valid_set_x_1, valid_set_x_2, valid_set_x_3, valid_set_x_4,[img_rows, img_cols])

    cnn1 = build_cnn(img_rows*2, img_cols*2)
    final_model = Sequential()
    final_model.add(cnn1)
    final_model.add(Dense(nb_classes, activation='softmax'))

    final_model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    full_model_adr = create_model_folder(__file__, save_address, verbose)

    loss_checker = ModelCheckpoint(full_model_adr + "/best_weights_loss.h5", monitor='val_loss',
                                   verbose=1, save_best_only=True, mode='auto', save_weights_only=True)
    acc_checker = ModelCheckpoint(full_model_adr + "/best_weights_acc.h5", monitor='val_acc',
                                  verbose=1, save_best_only=True, mode='auto', save_weights_only=True)

    if best_model_based_validset:
        callbacks = [acc_checker, loss_checker]
    else: callbacks = []

    final_model_adr = full_model_adr + '/final_model/'
    if not os.path.exists(final_model_adr):
        print('making... ' + final_model_adr)
        os.makedirs(final_model_adr)
    out_file = open(final_model_adr + '/out.txt', "w")
    out_file.write(pickle_adr + '\n')

    if all_data_for_train_flag:
        print(concat_train.shape[0], 'train samples')
        print(concat_valid.shape[0], 'validation samples')
        print(concat_test.shape[0], 'test samples')

        all_data_for_train = np.concatenate((concat_train, concat_valid), axis=0)
        all_data_for_train = np.concatenate((all_data_for_train, concat_test), axis=0)
        all_label_for_train = np.concatenate((cat_train_set_y_1, cat_valid_set_y_1), axis=0)
        all_label_for_train = np.concatenate((all_label_for_train, cat_test_set_y_1), axis=0)

        final_model.fit(all_data_for_train, all_label_for_train,
                        batch_size=batch_size, epochs=nb_epoch, verbose=2,
                        validation_data=(concat_valid, cat_valid_set_y_1), callbacks=callbacks)

    if verbose:
        score = final_model.evaluate(concat_test, cat_test_set_y_1, verbose=0)

        print('Test accuracy (last):', score[1])
        out_file.write('\n * Test Accuracy (last): %0.4f%% \n' % score[1])

        score2 = final_model.evaluate(concat_valid, cat_valid_set_y_1, verbose=0)
        print('valid accuracy (last):', score2[1])

        score3 = final_model.evaluate(concat_train, cat_train_set_y_1, verbose=0)
        print('Train accuracy (last):', score3[1])

    final_model.save(final_model_adr + '/multi_view_classifier.h5')

    if best_model_based_validset:
        final_model.load_weights(full_model_adr + "/best_weights_acc.h5")
        score = final_model.evaluate(concat_test, cat_test_set_y_1, verbose=0)
        print('Test accuracy (acc):', score[1])
        out_file.write('\n * Test Accuracy (acc): %0.4f%% \n' % score[1])
        score2 = final_model.evaluate(concat_valid, cat_valid_set_y_1, verbose=0)
        print('valid accuracy (acc):', score2[1])
        score3 = final_model.evaluate(concat_train, cat_train_set_y_1, verbose=0)
        print('Train accuracy (acc):', score3[1])

        final_model.load_weights(full_model_adr + "/best_weights_loss.h5")
        score = final_model.evaluate(concat_test, cat_test_set_y_1, verbose=0)
        print('Test accuracy (loss):', score[1])
        out_file.write('\n * Test Accuracy (loss): %0.4f%% \n' % score[1])
        score2 = final_model.evaluate(concat_valid, cat_valid_set_y_1, verbose=0)
        print('valid accuracy (loss):', score2[1])
        score3 = final_model.evaluate(concat_train, cat_train_set_y_1, verbose=0)
        print('Train accuracy (loss):', score3[1])


if __name__ == "__main__":
   print('Start ...')
   #main(batch_size,nb_epoch,train_flag,pickle_adr,save_address,number_of_classes,verbose)
   nb_classes = 22
   save_best_model = 0
   early_stopping = 0

   # default values for batch size and number of iterations
   batch_size = 30
   nb_epoch = 15

   # whether to use all data for training the model (1) or just the train set for training the model (0)
   all_data_for_train_flag = 1

   # the path where the pickles are there (default)
   pickle_adr = '/home/sara/information_science_journal/MLforGravitySpyJournal_from_neda_pc/pickles/pickle_2017-05-25_res0.3/'

   # the path where the trained model is saved
   save_address = './multi_view_models/alldata5/'
   save_best_model = 0
   image_size = [140, 170]
   verbose = 0

   main(batch_size, nb_epoch, all_data_for_train_flag, pickle_adr, save_address,
        nb_classes, image_size, save_best_model, verbose)
   print('Done!')

