""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Conv2D

import numpy as np

#4/2/2018
def concatenate_views(image_set1, image_set2, image_set3,
                      image_set4, image_size):
    """Create a merged view from a set of 4 views of one sample image

    Parameters:
        image_set1 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_set2 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_set3 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_set4 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_size (list):
            This refers to the shape of the non flattened pixelized image
            array

    Returns:
        concated_view (array):
            A single merged view of the sample, in the case
            a merged view of the 0.5 1.0 2.0 and 4.0 duration
            omega scans.
    """
    img_rows = image_size[0]
    img_cols = image_size[1]

    assert len(image_set1) == len(image_set2)
    concat_images_set = np.zeros((len(image_set1), 1, img_rows * 2, img_cols))
    for i in range(0, len(image_set1)):
        concat_images_set[i, :, :, :] = np.append(image_set1[i, :, :, :], image_set2[i, :, :, :], axis=1)

    assert len(image_set3) == len(image_set4)
    concat_images_set2 = np.zeros((len(image_set3), 1, img_rows * 2, img_cols))
    for i in range(0, len(image_set3)):
        concat_images_set2[i, :, :, :] = np.append(image_set3[i, :, :, :], image_set4[i, :, :, :], axis=1)

    out = np.append(concat_images_set,concat_images_set2, axis=3)
    return out

#4/2/2018
def build_cnn(img_rows, img_cols):
    """This is where we use Keras to build a covolutional neural network (CNN)

    The CNN built here is described in the
    `Table 5 <https://www.sciencedirect.com/science/article/pii/S0020025518301634#tbl0004>`_ 

    There are 5 layers. For each layer the logic is as follows

    input 2D matrix --> 2D Conv layer with x number of kernels with a 5 by 5 shape
    --> activation layer we use `ReLU <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_
    --> MaxPooling of size 2 by 2 all pixels are now grouped into bigger pixels of
    size 2 by 2 and the max pixel of the pixels that make up the 2 by 2 pizels is the
    value of the bigger pixel --> Dropout set to 50 percent. This means each pixel
    at this stage has a 50 percent chance of being set to 0.

    Parameters:
        image_rows (int):
            This refers to the number of rows in the non-flattened image

        image_cols (int):
            This refers to the number of cols in the non-flattened image

    Returns:
        model (`object`):
            a CNN
    """
    W_reg = 1e-4
    print('regularization parameter: ', W_reg)
    model = Sequential()
    model.add(Conv2D(16, (5, 5), padding='valid',
              input_shape=(1, img_rows, img_cols),
              kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(32, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(W_reg)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    print (model.summary())
    return model
