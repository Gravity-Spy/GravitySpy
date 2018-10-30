import keras.backend as K
K.set_image_data_format("channels_first")
from GS_utils import (cosine_distance, siamese_acc, eucl_dist_output_shape,
                      create_pairs3_gen)
from keras import regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda
from keras.models import Model,load_model
from keras.optimizers import RMSprop

from gravityspy.utils import log
from .read_image import read_rgb

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
                image_data_r, image_data_g, image_data_b  = read_rgb(os.path.join(path_to_trainingset,
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
               multi_view=True,
               batch_size=22, nb_epoch=10,
               nb_classes=22, image_size=[140, 170],
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
    reglularization = 1e-4
    logger = log.Logger('Gravity Spy: Training '
                        'Semantic Index')

    logger.info('Using random seed {0}'.format(random_seed))
    numpy.random.seed(random_seed)  # for reproducibility

    logger.info('You data set contained {0} samples'.format(len(data)))

    img_rows, img_cols = image_size[0], image_size[1]

    logger.info('The size of the images being trained {0}'.format(image_size))

    known_df = data.loc[~data.Label.isin(unknown_classes_labels)]
    known_df = known_df.groupby('Label').apply(lambda x: x.sample(frac=0.1,random_state=random_seed)).reset_index(drop=True).reset_index()
    known_df = known_df.sample(frac=1, random_state=random_seed)

    logger.info('Given unknown images here is what is to be considered in '
                'the known '
                'domain of samples which are {0}'.format(known_df.Label.unique()))

    logger.info('Selecting images to be considered in the unknown '
                'domain of samples which are {0}'.format(unknown_classes_labels))
    unknown_df = data.loc[data.Label.isin(unknown_classes_labels)]
    unknown_df = unknown_df.groupby('Label').apply(lambda x: x.sample(frac=0.1,random_state=random_seed)).reset_index(drop=True).reset_index()
    unknown_df = unknown_df.sample(frac=1, random_state=random_seed)

    known_x_1 = numpy.vstack(known_df['0.5.png'].values).reshape(
                                                     -1, 3, img_rows, img_cols)
    unknown_x_1 = numpy.vstack(unknown_df['0.5.png'].values).reshape(
                                                     -1, 3, img_rows, img_cols)

    known_x_2 = numpy.vstack(known_df['1.0.png'].values).reshape(
                                                     -1, 3, img_rows, img_cols)
    unknown_x_2 = numpy.vstack(unknown_df['1.0.png'].values).reshape(
                                                     -1, 3, img_rows, img_cols)

    known_x_3 = numpy.vstack(known_df['2.0.png'].values).reshape(
                                                     -1, 3, img_rows, img_cols)
    unknown_x_3 = numpy.vstack(unknown_df['2.0.png'].values).reshape(
                                                     -1, 3, img_rows, img_cols)

    known_x_4 = numpy.vstack(known_df['4.0.png'].values).reshape(
                                                     -1, 3, img_rows, img_cols)
    unknown_x_4 = numpy.vstack(unknown_df['4.0.png'].values).reshape(
                                                     -1, 3, img_rows, img_cols)

    if multi_view:
        known_classes = concatenate_views(known_x_1, known_x_2,
                            known_x_3, known_x_4, [img_rows, img_cols], True)
        unknown_classes = concatenate_views(unknown_x_1, unknown_x_2,
                            unknown_x_3, unknown_x_4, [img_rows, img_cols], True)
    else:
        # We are only using one duration for the similarity search
        known_classes = known_x_2
        unknown_classes = unknown_x_2

    # create binary pairs
    all_pairs_known = numpy.array(list(combinations(known_df['index'],2)))
    same_pairs_known = numpy.concatenate(known_df.groupby('Label')['index'].apply(lambda x : list(combinations(x.values,2))))
    s_pairs_known = set(map(tuple, same_pairs_known))
    a_pairs_known = set(map(tuple, all_pairs_known))
    diff_pairs_known = numpy.array(list(a_pairs_known-s_pairs_known))

    tmp1 = numpy.ones((same_pairs_known.shape[0], 3))
    tmp1[:,:-1] = same_pairs_known
    tmp2 = numpy.zeros((diff_pairs_known.shape[0], 3))
    tmp2[:,:-1] = diff_pairs_known
    pair_labels_known = numpy.vstack([tmp1, tmp2])

    all_pairs_unknown = numpy.array(list(combinations(unknown_df['index'],2)))
    same_pairs_unknown = numpy.concatenate(unknown_df.groupby('Label')['index'].apply(lambda x : list(combinations(x.values,2))))
    s_pairs_unknown = set(map(tuple, same_pairs_unknown))
    a_pairs_unknown = set(map(tuple, all_pairs_unknown))
    diff_pairs_unknown = numpy.array(list(a_pairs_unknown-s_pairs_unknown))

    tmp1 = numpy.ones((same_pairs_unknown.shape[0], 3))
    tmp1[:,:-1] = same_pairs_unknown
    tmp2 = numpy.zeros((diff_pairs_unknown.shape[0], 3))
    tmp2[:,:-1] = diff_pairs_unknown
    pair_labels_unknown = numpy.vstack([tmp1, tmp2])

    def generate_pairs(image_data, pair_labels, batch_size):
        pairs1 = []
        pairs2 = []
        labels = []
        counter = 0
        while True:
            for pair in pair_labels:
                counter += 1
                pairs1.append(known_classes[pair[0]])
                pairs2.append(known_classes[pair[1]])
                labels.append(pair[2])
                if counter == batch_size:
                    #yield np.array(pairs), np.array(labels)
                    yield [numpy.asarray(pairs1, np.float32), numpy.asarray(pairs2, np.float32)],numpy.asarray(labels, np.int32)
                    counter = 0
                    pairs1 = []
                    pairs2 = []
                    labels = []

    train_generator = generate_pairs(known_classes, pair_labels_known, batch_size)
    valid_generator = generate_pairs(unknown_classes, pair_labels_unknown, batch_size)

    # Create the model
    vgg16 = VGG16(weights='imagenet', include_top=False,
                  input_shape=known_classes.shape[1:])
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

    train_negative_factor = 1
    test_negative_factor = 1
    # the samples from unknown_classes should be separated for test and valid in future

    train_batch_num = (len(known_classes) * (train_negative_factor + 1)) / batch_size
    logger.info('train batch num {0}'.format(train_batch_num))
    valid_batch_num = (len(unknown_classes) * (test_negative_factor + 1)) / batch_size

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
