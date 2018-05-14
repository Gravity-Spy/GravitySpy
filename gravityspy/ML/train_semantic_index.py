import keras.backend as K
from GS_utils import cosine_distance
from keras import regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda
from keras.models import Model,load_model
from keras.optimizers import RMSprop

from gravityspy.utils import log
import numpy as np

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
    np.random.seed(random_seed)  # for reproducibility

    logger.info('Loading file {0}'.format(data))

    image_dataDF = pd.read_pickle(data)

    logger.info('You data set contained {0} samples'.format(len(image_dataDF)))

    img_rows, img_cols = image_size[0], image_size[1]

    logger.info('The size of the images being trained {0}'.format(image_size))

    logger.info('Selecting images to be considered in the known '
                'domain of samples which are {0}'.format(image_dataDF.Label.unique()))
    knownDF = image_dataDF.loc[~image_dataDF.Label.isin(
                                          unknown_classes_labels)]
    logger.info('Selecting images to be considered in the unknown '
                'domain of samples which are {0}'.format(unknown_classes_labels))
    unknownDF = image_dataDF.loc[image_dataDF.isin(unknown_classes_labels)]

    known_x_1 = np.vstack(knownDF['0.5.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
    unknownn_x_1 = np.vstack(unknownDF['0.5.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)

    known_x_2 = np.vstack(knownDF['1.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
    unknownn_x_2 = np.vstack(unknownDF['1.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)

    known_x_3 = np.vstack(knownDF['2.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
    unknown_x_3 = np.vstack(unknownDF['2.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)

    known_x_4 = np.vstack(knownDF['4.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
    unknown_x_4 = np.vstack(unknownDF['4.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)

    if multi_view:
        known_classes = concatenate_views(known_set_x_1, known_set_x_2,
                            known_set_x_3, known_set_x_4, [img_rows, img_cols])
        unknown_classes = concatenate_views(unknown_set_x_1, unknown_set_x_2,
                            unknown_set_x_3, unknown_set_x_4, [img_rows, img_cols])
    else:
        # We are only using one duration for the similarity search
        known_classes_image_array = known_set_x_2
        unknown_classes_image_array = unknown_set_x_2

    # Generate the Binary pairs for training.
    train_generator = create_pairs3_gen(data_, known_classes_indices_for_metric_learning, size_of_batch)
    valid_generator = create_pairs3_gen(data_x_2, unknown_classes_indices_for_clustering, size_of_batch)

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
    train_generator = create_pairs3_gen(data_, known_classes_indices_for_metric_learning, size_of_batch)
    valid_generator = create_pairs3_gen(data_x_2, unknown_classes_indices_for_clustering, size_of_batch)
    # the samples from data_x_2 should be separated for test and valid in future

    train_batch_num = (len(data_x_1) * (train_negative_factor + 1)) / size_of_batch
    logger.info('train batch num {0}'.format(train_batch_num))
    valid_batch_num = (len(data_x_2) * (test_negative_factor + 1)) / size_of_batch

    #similarity_model.fit_generator(train_generator, validation_data=valid_generator, verbose=2,
     #                   steps_per_epoch=train_batch_num, validation_steps=valid_batch_num, epochs=nb_epoch)
    #train_batch_num = 10
    #valid_batch_num = 10
    similarity_model.fit_generator(train_generator, validation_data=valid_generator, verbose=2,
                        steps_per_epoch=train_batch_num, validation_steps=valid_batch_num, epochs=nb_epoch)

    # validation
    logger.info('validating the model')
    out_file.write('validating the model ...' + '\n')

    '''known_classes_negative_factor = 1
    unknown_classes_negative_factor = 1
    known_classes_generator = create_pairs3_gen(data_x_1, known_classes_indices_for_metric_learning, size_of_batch)
    unknown_classes_generator = create_pairs3_gen(data_x_2, unknown_classes_indices_for_clustering, size_of_batch)
    known_classes_batch_num = (len(data_x_1) * (known_classes_negative_factor + 1)) / size_of_batch
    unknown_classes_batch_num = (len(data_x_2) * (unknown_classes_negative_factor + 1)) / size_of_batch'''

    logger.info('Known classes')
    out_file.write('known classes' + '\n')
    #res1 = similarity_model.evaluate_generator(known_classes_generator, known_classes_batch_num)
    res1 = similarity_model.evaluate_generator(train_generator, train_batch_num)
    logger.info(res1)
    for item in res1:
            out_file.write("%f, " % item)

    logger.info(' unknown classes')
    out_file.write('unknown classes' + '\n')
    #res2 = similarity_model.evaluate_generator(unknown_classes_generator, unknown_classes_batch_num)
    res2 = similarity_model.evaluate_generator(valid_generator, valid_batch_num)
    logger.info(res2)
    for item in res2:
            out_file.write("%f, " % item)

    similarity_model.compile(loss='mean_squared_error', optimizer='rmsprop')
    similarity_model.save(os.join.path(model_adr, 'similarity_metric_model.h5'))
    semantic_idx_model.save(os.join.path(model_adr, 'semantic_idx_model.h5'))
