from GS_utils import build_cnn, my_load_dataset
from GS_utils import create_model_folder, concatenate_views
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import sys, getopt
import gzip, os
import cPickle
import make_pickle_for_linux as make_pickle
from gravityspy.utils import log
import pandas as pd

'''
By Sara Bahaadini
This function reads the  pickle files of golden_set and train a ML classifier
and write it into a model folder
'''

def pickle_trainingset(path_to_trainingset, save_address='pickleddata',
                       verbose=False):
    """Parameters
    ----------
    path_to_trainingset : `str` pwd to trainingset
                 where format of training set folder
                 is "somedirectoryname"/"classname/"images"
    save_address : `str` path to folder you would like to save
                 the pixelated training data
                 optional Default `pickleddata`
    verbose : `boolean`, default False

    Returns
    -------
    A pickled `pandas.DataFrame` with rows of samples
    and columns containing the pixelated 0.5, 1.0, 2.0,
    and 4.0 duration images as well as a column with the True
    Label and a column with an ID that uniquely identifies that sample
    """

    logger = log.Logger('Gravity Spy: Pickling '
                        'Trainingset')

    if not os.path.exists(save_address):
        if verbose:
            logger.info('making... ' + save_address)
        os.makedirs(save_address)

    classes = sorted(os.listdir(path_to_trainingset))
    nb_classes = len(classes)
    logger.info('The number of classes are {0}'.format(nb_classes))
    logger.info('The classes you are pickling are {0}'.format(
          classes))

    image_dataDF = pd.DataFrame()
    for iclass in classes:
        logger.info('Converting {0} into b/w info'.format(iclass))
        images = sorted(os.listdir(os.path.join(path_to_trainingset, iclass)))
        images = [imageidx for imageidx in images \
                  if 'L1_' in imageidx or 'H1_' in imageidx]
        # Group each sample into sets of 4 different durations
        samples = zip(*(iter(images),) * 4)
        for isample in samples:
            tmpDF = pd.DataFrame()
            for idur in isample:
                if verbose:
                    logger.info('Converting {0}'.format(idur))
                image_data = make_pickle.main(os.path.join(path_to_trainingset, 
                                              iclass, idur), resolution=0.3)
                information_on_image = idur.split('_')
                tmpDF[information_on_image[-1]] = [image_data]
            tmpDF['uniqueID'] = information_on_image[1]
            tmpDF['Label'] = iclass
            image_dataDF = image_dataDF.append(tmpDF)

        logger.info('Finished converting {0} into b/w info'.format(iclass))

    picklepath = os.path.join(save_address, 'trainingset.pkl')
    logger.info('Saving pickled data to {0}'.format(picklepath))
    image_dataDF.to_pickle(picklepath)
    return image_dataDF


def make_model(data, model_folder='model', batch_size=22, nb_epoch=10,
               nb_classes=22, fraction_validation=.125, fraction_testing=None,
               best_model_based_validset=0, image_size=[140, 170],
               random_seed=1986, verbose=True):
    """Parameters
    ----------
    data : `str` file path to pickled trianing set data
    model_folder : `str` path to folder you would like to save
                 the model
                 optional Default `model`
    batch_size : `int` default 22
    nb_epoch : `int` default 40
    nb_classes : `int` default 22
    fraction_validation : `float` default .125 
    fraction_testing : `float` default None
    all_data_for_train_flag : `int` default 1
    best_model_based_validset : `int` default 0
    image_size : `int` default [140, 170]
    random_seed : `int`  default 1986
    verbose : `boolean`, default False

    Returns
    -------
    A training Convultional Neural Network
    """
    logger = log.Logger('Gravity Spy: Training '
                        'Model')

    logger.info('Using random seed {0}'.format(random_seed))
    np.random.seed(random_seed)  # for reproducibility

    logger.info('Loading file {0}'.format(data))

    image_dataDF = pd.read_pickle(data)

    logger.info('You data set contained {0} samples'.format(len(image_dataDF)))

    img_rows, img_cols = image_size[0], image_size[1]

    logger.info('The size of the images being trained {0}'.format(image_size))

    classes = sorted(image_dataDF.Label.unique())

    if len(classes) != nb_classes:
        raise ValueError('Youre supplied data set does not match the number of'
                         ' classes you said you were training on')

    classes = dict(enumerate(classes))
    classes = dict((str(v),k) for k,v in classes.iteritems())

    logger.info('You have supplied a training set with the following class'
                'idx to str label mapping: {0}'.format(classes))

    logger.info('converting string to idx...')
    image_dataDF.Label = image_dataDF.Label.apply(lambda x: classes[x])

    logger.info('Selecting samples for validation ...')
    logger.info('You have selected to set aside {0} percent of '
                'images per class for validation.'.format(
                                       fraction_validation * 100))

    validationDF = image_dataDF.groupby('Label').apply(
                       lambda x: x.sample(frac=fraction_validation,
                       random_state=random_seed)
                       ).reset_index(drop=True)

    logger.info('Removing validation images from training DF ...')

    image_dataDF = image_dataDF.loc[~image_dataDF.uniqueID.isin(
                                    validationDF.uniqueID)]

    logger.info('There are now {0} samples remaining'.format(
                                                       len(image_dataDF)))

    if fraction_testing:
        logger.info('Selecting samples for testing ...')
        logger.info('You have selected to set aside {0} percent of ' 
                'images per class for validation.'.format(

                                       fraction_testing * 100))
        testingDF = image_dataDF.groupby('Label').apply(
                   lambda x: x.sample(frac=fraction_testing,
                             random_state=random_seed)
                   ).reset_index(drop=True)

        logger.info('Removing testing images from training DF ...')
        
        image_dataDF = image_dataDF.loc[~image_dataDF.uniqueID.isin(
                                        testingDF.uniqueID)]
        
        logger.info('There are now {0} samples remaining'.format(
                                                           len(image_dataDF)))

    # concatenate the pixels
    train_set_x_1 = np.vstack(image_dataDF['0.5.png'].as_matrix())
    validation_x_1 = np.vstack(validationDF['0.5.png'].as_matrix())

    train_set_x_2 = np.vstack(image_dataDF['1.0.png'].as_matrix())
    validation_x_2 = np.vstack(validationDF['1.0.png'].as_matrix())

    train_set_x_3 = np.vstack(image_dataDF['2.0.png'].as_matrix())
    validation_x_3 = np.vstack(validationDF['2.0.png'].as_matrix())

    train_set_x_4 = np.vstack(image_dataDF['4.0.png'].as_matrix())
    validation_x_4 = np.vstack(validationDF['4.0.png'].as_matrix())

    train_set_x_1 = train_set_x_1.reshape(-1, 1, img_rows, img_cols)
    validation_x_1 = validation_x_1.reshape(-1, 1, img_rows, img_cols)

    train_set_x_2 = train_set_x_2.reshape(-1, 1, img_rows, img_cols)
    validation_x_2 = validation_x_2.reshape(-1, 1, img_rows, img_cols)

    train_set_x_3 = train_set_x_3.reshape(-1, 1, img_rows, img_cols)
    validation_x_3 = validation_x_3.reshape(-1, 1, img_rows, img_cols)

    train_set_x_4 = train_set_x_4.reshape(-1, 1, img_rows, img_cols)
    validation_x_4 = validation_x_4.reshape(-1, 1, img_rows, img_cols)

    if fraction_testing:
        testing_x_1 = np.vstack(testingDF['0.5.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_2 = np.vstack(testingDF['1.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_3 = np.vstack(testingDF['2.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_4 = np.vstack(testingDF['4.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)

    # Concatenate the labels
    trainingset_labels = np.vstack(image_dataDF['Label'].as_matrix())
    validation_labels = np.vstack(validationDF['Label'].as_matrix())
    if fraction_testing:
        testing_labels = np.vstack(testingDF['Label'].as_matrix())

    # Concatenate the name
    trainingset_names = np.vstack(image_dataDF['uniqueID'].as_matrix())
    validation_names = np.vstack(validationDF['uniqueID'].as_matrix())
    if fraction_testing:
        testing_names = np.vstack(testingDF['uniqueID'].as_matrix())

    # Categorize labels
    trainingset_labels = np_utils.to_categorical(
                                        trainingset_labels, nb_classes)

    validation_labels = np_utils.to_categorical(
                                        validation_labels, nb_classes)
    if fraction_testing:
        testing_labels = np_utils.to_categorical(
                                        testing_labels, nb_classes)

    logger.info('Concatenating multiple views ...')
    concat_train = concatenate_views(train_set_x_1, train_set_x_2,
                            train_set_x_3, train_set_x_4, [img_rows, img_cols])
    concat_valid = concatenate_views(validation_x_1, validation_x_2,
                            validation_x_3, validation_x_4,
                            [img_rows, img_cols])

    if fraction_testing:
        concat_test = concatenate_views(testing_x_1, testing_x_2,
                            testing_x_3, testing_x_4,
                            [img_rows, img_cols])

    cnn1 = build_cnn(img_rows*2, img_cols*2)
    final_model = Sequential()
    final_model.add(cnn1)
    final_model.add(Dense(nb_classes, activation='softmax'))

    final_model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    if not os.path.exists(model_folder):
        if verbose:
            logger.info('making... ' + model_folder)
        os.makedirs(model_folder)

    full_model_adr = model_folder

    loss_checker = ModelCheckpoint(os.path.join(full_model_adr,
                                                "best_weights_loss.h5"),
                                   monitor='val_loss', verbose=1,
                                   save_best_only=True, mode='auto',
                                   save_weights_only=True)
    acc_checker = ModelCheckpoint(os.path.join(full_model_adr,
                                               "best_weights_acc.h5"),
                                  monitor='val_acc', verbose=1,
                                  save_best_only=True, mode='auto',
                                  save_weights_only=True)

    if best_model_based_validset:
        callbacks = [acc_checker, loss_checker]
    else:
        callbacks = []

    final_model_adr = os.path.join(full_model_adr, 'final_model')
    if not os.path.exists(final_model_adr):
        logger.info('making... ' + final_model_adr)
        os.makedirs(final_model_adr)
    out_file = open(os.path.join(final_model_adr, 'out.txt'), "w")
    out_file.write(data + '\n')


    final_model.fit(concat_train, trainingset_labels,
        batch_size=batch_size, epochs=nb_epoch, verbose=1,
        validation_data=(concat_valid, validation_labels), callbacks=callbacks)

    if fraction_testing:
        score = final_model.evaluate(concat_test, testing_labels, verbose=0)

        logger.info('Test accuracy (last): {0}'.format(score[1]))
        out_file.write('\n * Test Accuracy (last): %0.4f%% \n' % score[1])

        score2 = final_model.evaluate(concat_valid, validation_labels, verbose=0)
        logger.info('valid accuracy (last): {0}'.format(score2[1]))

        score3 = final_model.evaluate(concat_train, trainingset_labels, verbose=0)
        logger.info('Train accuracy (last): {0}'.format(score3[1]))

    else:
        score2 = final_model.evaluate(concat_valid, validation_labels, verbose=0)
        logger.info('valid accuracy (last): {0}'.format(score2[1]))

        score3 = final_model.evaluate(concat_train, trainingset_labels, verbose=0)
        logger.info('Train accuracy (last): {0}'.format(score3[1]))

    final_model.save(os.path.join(final_model_adr, 'multi_view_classifier.h5'))

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

        return


def main(batch_size, nb_epoch, all_data_for_train_flag, pickle_adr, save_address,
         number_of_classes, best_model_based_validset, image_size=[140, 170], verbose=True):

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
   nb_classes = 22
   save_best_model = 0
   early_stopping = 0

   batch_size = 30
   nb_epoch = 40

   # whether to use all data for training the model (1) or just the train set for training the model (0)
   all_data_for_train_flag = 1

   # the path where the pickles are there (default)
   pickle_adr = '/home/sara/information_science_journal/MLforGravitySpyJournal_from_neda_pc/pickles/pickle_2017-05-25_res0.3/'

   # the path where the trained model is saved
   save_address = './multi_view_models/alldata5/'
   save_best_model = 0
   image_size = [140, 170]
   verbose = 1

   main(batch_size, nb_epoch, all_data_for_train_flag, pickle_adr, save_address,
        nb_classes, save_best_model, image_size, verbose)
   print('Done!')

