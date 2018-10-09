import keras.backend as K
K.set_image_data_format("channels_first")
from GS_utils import cosine_distance
from keras import regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda
from keras.models import Model,load_model
from keras.optimizers import RMSprop

from gravityspy.utils import log
import numpy
import os
import pandas

def pickle_trainingset(path_to_trainingset,
                       save_address='pickleddata/trainingset.pkl',
                       verbose=False):
    """Pre-processes the training set images and save to pickle.

    Parameters:

        path_to_trainingset (str):
            Path to trainingset where format of training set folder
            is "somedirectoryname"/"classname/"images"

        save_address (str, optional):
            Defaults to `pickleddata`
            Path to folder you would like to save the pixelated training data

        verbose (bool, optional):
            Defaults to False
            Extra verbosity

    Returns:

        A pickled `pandas.DataFrame`:
            with rows of samples
            and columns containing the pixelated 0.5, 1.0, 2.0,
            and 4.0 duration images as well as a column with the True
            Label and a column with an ID that uniquely identifies that sample
    """

    logger = log.Logger('Gravity Spy: Pickling '
                        'Trainingset RGB')

    if not os.path.exists(os.path.dirname(save_address)):
        if verbose:
            logger.info('making... ' + os.path.dirname(save_address))
        os.makedirs(os.path.dirname(save_address))

    classes = sorted(os.listdir(path_to_trainingset))
    nb_classes = len(classes)
    logger.info('The number of classes are {0}'.format(nb_classes))
    logger.info('The classes you are pickling are {0}'.format(
          classes))

    data = pandas.DataFrame()
    for iclass in classes:
        logger.info('Converting {0} into RGB info'.format(iclass))
        images = sorted(os.listdir(os.path.join(path_to_trainingset, iclass)))
        images = [imageidx for imageidx in images \
                  if 'L1_' in imageidx or 'H1_' in imageidx or 'V1_' in imageidx]
        # Group each sample into sets of 4 different durations
        samples = zip(*(iter(images),) * 4)
        for isample in samples:
            tmpdf = pandas.DataFrame()
            for idur in isample:
                if verbose:
                    logger.info('Converting {0}'.format(idur))
                image_data_r, image_data_g, image_data_b  = read_image.read_rgb(os.path.join(path_to_trainingset,
                                                              iclass, idur),
                                                              resolution=0.3)
                information_on_image = idur.split('_')
                tmpdf[information_on_image[-1]] = [[image_data_r, image_data_g, image_data_b]]
            tmpdf['uniqueID'] = information_on_image[1]
            tmpdf['Label'] = iclass
            data = data.append(tmpdf)

        logger.info('Finished converting {0} into b/w info'.format(iclass))

    picklepath = os.path.join(save_address)
    logger.info('Saving pickled data to {0}'.format(picklepath))
    data.to_pickle(picklepath)
    return data

def make_model(data, model_folder='model',
               unknown_classes_labels=['Whistle', 'Scratchy'],
               multi_view=False,
               batch_size=22, nb_epoch=10,
               nb_classes=22, fraction_validation=.125, fraction_testing=None,
               best_model_based_validset=0, image_size=[140, 170],
               random_seed=1986, verbose=True):
    """Train a Semantic Index.

    This module uses `keras <https://keras.io/>`_ to interface
    with the creation of a neural net and currently uses
    `theano <http://deeplearning.net/software/theano/>`_ as the backend
    for doing the heavy gpu lifting.

    The optimizer :class:`keras.optimizers.Adadelta`

    The loss function being optimized is `softmax <https://en.wikipedia.org/wiki/Softmax_function>`_

    Parameters:
        data (str):
            Pickle file containing training set data

        model_folder (str, optional):
            Defaults to `model`
            path to folder you would like to save the model

        unknown_classes_labels (list, optional):
            Defaults to ['Whistle', 'Scratchy']
            A list of classes to be considered as the unknown
            domain for which clustering will be performed
            once the knowledge from the known classes
            has been trained.

        batch_size (int, optional):
            Default 22

        nb_epoch (int, optional):
            Default 40

        nb_classes (int, optional):
            Default 22

        fraction_validation (float, optional):
            Default .125

        fraction_testing (float, optional):
            Default None

        all_data_for_train_flag (int, optional):
            Default 1

        best_model_based_validset (int,optional):
    """
    logger = log.Logger('Gravity Spy: Training '
                        'Semantic Index')

    logger.info('Using random seed {0}'.format(random_seed))
    numpy.random.seed(random_seed)  # for reproducibility

    logger.info('You data set contained {0} samples'.format(len(data)))

    img_rows, img_cols = image_size[0], image_size[1]

    logger.info('The size of the images being trained {0}'.format(image_size))

    known_df = data.loc[~data.Label.isin(unknown_classes_labels)]

    logger.info('Given unknown images here is what is to be considered in '
                'the known '
                'domain of samples which are {0}'.format(knwon_df.Label.unique()))

    logger.info('Selecting images to be considered in the unknown '
                'domain of samples which are {0}'.format(unknown_classes_labels))
    unknown_df = data.loc[data.Label.isin(unknown_classes_labels)]

    known_x_1 = numpy.vstack(known_df['0.5.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    unknownn_x_1 = numpy.vstack(unknown_df['0.5.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    known_x_2 = numpy.vstack(known_df['1.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    unknownn_x_2 = numpy.vstack(unknown_df['1.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    known_x_3 = numpy.vstack(known_df['2.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    unknown_x_3 = numpy.vstack(unknown_df['2.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    known_x_4 = numpy.vstack(known_df['4.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    unknown_x_4 = numpy.vstack(unknown_df['4.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    test_set_unlabelled_x_1 = image_data.filter(regex=("1.0.png")).iloc[0].iloc[0]
    test_set_unlabelled_x_2 = image_data.filter(regex=("2.0.png")).iloc[0].iloc[0]
    test_set_unlabelled_x_3 = image_data.filter(regex=("4.0.png")).iloc[0].iloc[0]
    test_set_unlabelled_x_4 = image_data.filter(regex=("0.5.png")).iloc[0].iloc[0]
    test_set_unlabelled_x_1 = numpy.concatenate((test_set_unlabelled_x_1[0].reshape(-1, 1, img_rows, img_cols),
                                              test_set_unlabelled_x_1[1].reshape(-1, 1, img_rows, img_cols),
                                              test_set_unlabelled_x_1[2].reshape(-1, 1, img_rows, img_cols)),
                                             axis=1)
    test_set_unlabelled_x_2 = numpy.concatenate((test_set_unlabelled_x_2[0].reshape(-1, 1, img_rows, img_cols),
                                              test_set_unlabelled_x_2[1].reshape(-1, 1, img_rows, img_cols),
                                              test_set_unlabelled_x_2[2].reshape(-1, 1, img_rows, img_cols)),
                                             axis=1)
    test_set_unlabelled_x_3 = numpy.concatenate((test_set_unlabelled_x_3[0].reshape(-1, 1, img_rows, img_cols),
                                              test_set_unlabelled_x_3[1].reshape(-1, 1, img_rows, img_cols),
                                              test_set_unlabelled_x_3[2].reshape(-1, 1, img_rows, img_cols)),
                                             axis=1)
    test_set_unlabelled_x_4 = numpy.concatenate((test_set_unlabelled_x_4[0].reshape(-1, 1, img_rows, img_cols),
                                              test_set_unlabelled_x_4[1].reshape(-1, 1, img_rows, img_cols),
                                              test_set_unlabelled_x_4[2].reshape(-1, 1, img_rows, img_cols)),
                                             axis=1)

    if multi_view:
        known_classes = concatenate_views(known_set_x_1, known_set_x_2,
                            known_set_x_3, known_set_x_4, [img_rows, img_cols], True)
        unknown_classes = concatenate_views(unknown_set_x_1, unknown_set_x_2,
                            unknown_set_x_3, unknown_set_x_4, [img_rows, img_cols], True)
    else:
        # We are only using one duration for the similarity search
        known_classes_image_array = known_set_x_2
        unknown_classes_image_array = unknown_set_x_2

    # Generate the Binary pairs for training.
    train_generator = create_pairs3_gen(known_classes, known_classes_indices_for_metric_learning, size_of_batch)
    valid_generator = create_pairs3_gen(unknown_classes, unknown_classes_indices_for_clustering, size_of_batch)

    # Create the model
    vgg16 = VGG16(weights='imagenet', include_top=False,
                  input_shape=data_x_1.shape[1:])
    x = vgg16.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, kernel_regularizer=regularizers.l2(reglularization))(x)
    predictions = Dense(200, activation='relu')(x)


    #Then create the corresponding model
    base_network = Model(input=vgg16.input, output=predictions)

    if multi_view:
        img_cols = 2*img_cols
        img_rows = 2*img_rows

    input_a = Input(shape=(3, img_rows, img_cols))
    input_b = Input(shape=(3, img_rows, img_cols))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(cosine_distance,
                      output_shape=eucl_dist_output_shape)(
                          [processed_a, processed_b]
                      )

    similarity_model = Model(inputs=[input_a, input_b], outputs=distance)
    semantic_idx_model = Model(inputs=[input_a], outputs=processed_a)

    number_of_cnn_from_vgg = 0
    for i in range(len(vgg16.layers)-number_of_cnn_from_vgg):
        vgg16.layers[i].trainable = False


    similarity_model.summary()
    semantic_idx_model.summary()
    rms = RMSprop()

    similarity_model.compile(loss=contrastive_loss, optimizer=rms,
                             metrics=[siamese_acc(0.3), siamese_acc(0.4),
                                      siamese_acc(0.5), siamese_acc(0.6),
                                      siamese_acc(0.7), siamese_acc(0.8),
                                      siamese_acc(0.9), siamese_acc(0.925),
                                      siamese_acc(0.95), siamese_acc(0.975),
                                      siamese_acc(0.985),siamese_acc(0.99)])

    # train
    logger.info('training the model ...')
    out_file.write('training the model ...' + '\n')

    train_negative_factor = 1
    test_negative_factor = 1
    # the samples from data_x_2 should be separated for test and valid in future

    train_batch_num = (len(data_x_1) * (train_negative_factor + 1)) / size_of_batch
    logger.info('train batch num {0}'.format(train_batch_num))
    valid_batch_num = (len(data_x_2) * (test_negative_factor + 1)) / size_of_batch

    similarity_model.fit_generator(train_generator, validation_data=valid_generator, verbose=2,
                        steps_per_epoch=train_batch_num, validation_steps=valid_batch_num, epochs=nb_epoch)

    # validation
    logger.info('validating the model')

    logger.info('Known classes')
    res1 = similarity_model.evaluate_generator(train_generator, train_batch_num)
    logger.info(res1)

    logger.info(' unknown classes')
    res2 = similarity_model.evaluate_generator(valid_generator, valid_batch_num)
    logger.info(res2)

    similarity_model.compile(loss='mean_squared_error', optimizer='rmsprop')
    similarity_model.save(os.join.path(model_adr, 'similarity_metric_model.h5'))
    semantic_idx_model.save(os.join.path(model_adr, 'semantic_idx_model.h5'))
