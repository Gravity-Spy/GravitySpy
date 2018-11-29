from .GS_utils import concatenate_views
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

import numpy
import os

'''
By Sara Bahaadini and Neda Rohani, IVPL, Northwestern University.
This function reads the trained ML classifier and pickle files of unlabelled glitches and generate the score
file in a .csv file
'''
def convert_pd_image_info_to_cnn_ready(image_data, reshape_order,
                                       rgb, order_of_channels,
                                       preprocess_input=False):
    """Convert pandasdf of image data to numpy array ready for CNN

    Parameters:

        image_data (`pd.DataFrame`):
            This is a DF with columns whose names are the same as
            the name of the image and whose row entries are
            a flatten numpy.array of the pixel values

        reshape_order (tuple):
            the correct way to reshape the flatten image arrays
            examples include: (-1, 3, img_rows, img_cols),
            (-1, img_rows, img_cols, 1), etc.

        rgb (bool):
            Are you trying to create a merged view of grayscale
            or RGB image renders

        preprocess_input (bool):
            If converting image data for the similarity
            feature space extraction, then the sample must
            go through a preprocessing step after creating the merged
            image.

    Returns:

        merged_image_data (np.array):
            returns a numpy array containing the numpy array
            configuration for the images. The order of the tiling
            from merged view is 0.5, 1.0, 2.0, 4.0

        ids (list):
            list of the unique ids that go with each merged image.
    """
    half_second_images = sorted(image_data.filter(regex=("0.5.png")).keys())
    one_second_images = sorted(image_data.filter(regex=("1.0.png")).keys())
    two_second_images = sorted(image_data.filter(regex=("2.0.png")).keys())
    four_second_images = sorted(image_data.filter(regex=("4.0.png")).keys())

    test_set_unlabelled_x_1 = numpy.vstack(image_data[half_second_images].iloc[0]).reshape(reshape_order)
    test_set_unlabelled_x_2 = numpy.vstack(image_data[one_second_images].iloc[0]).reshape(reshape_order)
    test_set_unlabelled_x_3 = numpy.vstack(image_data[two_second_images].iloc[0]).reshape(reshape_order)
    test_set_unlabelled_x_4 = numpy.vstack(image_data[four_second_images].iloc[0]).reshape(reshape_order)

    merged_image_data = concatenate_views(test_set_unlabelled_x_1,
                                          test_set_unlabelled_x_2,
                                          test_set_unlabelled_x_3,
                                          test_set_unlabelled_x_4,
                                          [img_rows, img_cols],
                                          rgb_flag=rgb,
                                          order_of_channels=order_of_channels)

     if preprocess_input:
        merged_image_data = preprocess_input(merged_image_data)

    ids = []
    for uid in half_second_images:
        ids.append(uid.split('_')[1])

    return merged_image_data, ids, half_second_images, one_second_images, two_second_images, four_second_images

def label_glitches(image_data, model_name, order_of_channels="channels_last",
                   image_size=[140, 170], verbose=False):
    """Obtain 1XNclasses confidence vector and label for image

    Parameters:

        image_data (`pd.DataFrame`):
            This is a DF with columns whose names are the same as
            the image and whose row entries are
            the b/w pixel values at some resoltion
            determined by `read_image`

        model_adr (str, optional):
            Path to folder containing model

        image_size (list, optional):
            Default [140, 170]

        verbose (bool, optional):
            Default False

    Returns:

        score3_unlabelled (np.array):
            confidence scores per class (b/t 0 and 1)
            index_label (int): ml label
    """
    numpy.random.seed(1986)  # for reproducibility
    img_rows, img_cols = image_size[0], image_size[1]

    K.set_image_data_format(order_of_channels)
    if order_of_channels == 'channels_last':
        reshape_order = (-1, img_rows, img_cols, 1)
    elif order_of_channels == 'channels_first':
        reshape_order = (-1, 1, img_rows, img_cols)
    else:
        raise ValueError("Do not understand supplied channel order")

    merged_image_data, ids, half_second_images, one_second_images, two_second_images, four_second_images =\
        convert_pd_image_info_to_cnn_ready(image_data,
                                           reshape_order,
                                           rgb=False,
                                           order_of_channels=order_of_channels,
                                           preprocess_input=False)

    # load a model and weights
    final_model = load_model(model_name)

    final_model.compile(loss='categorical_crossentropy',
                        optimizer='adadelta',
                        metrics=['accuracy'])

    confidence_array = final_model.predict_proba(merged_image_data, verbose=0)
    index_label = confidence_array.argmax(1)

    return confidence_array, index_label, ids, half_second_images, one_second_images, two_second_images, four_second_images

def get_multiview_feature_space(image_data, semantic_model_name,
                                order_of_channels="channels_last",
                                image_size=[140, 170], verbose=False):
    """Obtain N dimensional feature space of sample

    Parameters:

        image_data (`pd.DataFrame`):
            This is a DF with columns whose names are the same as
            the image and whose row entries are
            the b/w pixel values at some resoltion
            determined by `read_image`

        semantic_model_adr (str):
            Path to folder containing similarity model

        image_size (list, optional):
            default [140, 170]

        verbose (bool, optional):
            default False

    Returns:

        np.array:
            a 200 dimensional feature space vector
    """
    numpy.random.seed(1986)
    img_rows, img_cols = image_size[0], image_size[1]

    K.set_image_data_format(order_of_channels)
    if order_of_channels == 'channels_last':
        reshape_order = (-1, img_rows, img_cols, 3)
    elif order_of_channels == 'channels_first':
        reshape_order = (-1, 3, img_rows, img_cols)
    else:
        raise ValueError("Do not understand supplied channel order")

    merged_image_data, ids, _, _, _, _ =\
        convert_pd_image_info_to_cnn_ready(image_data,
                                           reshape_order,
                                           rgb=True,
                                           order_of_channels=order_of_channels,
                                           preprocess_input=True)

    semantic_idx_model = load_model(semantic_model_name)

    features = semantic_idx_model.predict([merged_image_data])

    return features, ids

def get_deeplayer(image_data, semantic_model_name,
                  order_of_channels="channels_last",
                  image_size=[140, 170], verbose=False):
    """Obtain 1XNclasses confidence vector and label for image

    Parameters:

        image_data (`pd.DataFrame`):
            This is a DF with columns whose names are the same as
            the image and whose row entries are
            the b/w pixel values at some resoltion
            determined by `read_image`

        model_adr (str, optional):
            Path to folder containing model

        image_size (list, optional):
            Default [140, 170]

        verbose (bool, optional):
            Default False

    Returns:

        score3_unlabelled (np.array):
            confidence scores per class (b/t 0 and 1)
            index_label (int): ml label
    """
    numpy.random.seed(1986)  # for reproducibility
    img_rows, img_cols = image_size[0], image_size[1]

    K.set_image_data_format(order_of_channels)
    if order_of_channels == 'channels_last':
        reshape_order = (-1, img_rows, img_cols, 1)
    elif order_of_channels == 'channels_first':
        reshape_order = (-1, 1, img_rows, img_cols)
    else:
        raise ValueError("Do not understand supplied channel order")

    merged_image_data, ids, half_second_images, one_second_images, two_second_images, four_second_images =\
        convert_pd_image_info_to_cnn_ready(image_data,
                                           reshape_order,
                                           rgb=False,
                                           order_of_channels=order_of_channels,
                                           preprocess_input=False)

    # load a model and weights
    final_model = load_model(model_name)

    final_model.compile(loss='categorical_crossentropy',
                        optimizer='adadelta',
                        metrics=['accuracy'])

    feature_exc1 = K.function([final_model.layers[0].get_input_at(node_index=0),
                K.learning_phase()],
               [final_model.layers[0].get_layer(index=20).output])

    deeplayer = feature_exc1([merged_image_data, 0])[0]
    confidence_array = final_model.predict_proba(merged_image_data, verbose=0)
    index_label = confidence_array.argmax(1)

    return confidence_array, index_label, deeplayer, ids, half_second_images, one_second_images, two_second_images, four_second_images
