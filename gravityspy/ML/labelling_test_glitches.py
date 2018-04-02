from GS_utils import load_dataset_unlabelled_glitches
import numpy as np
from keras.utils import np_utils
from functions_fusion import square_early_concatenate_feature
import sys, gzip, cPickle, os
from getopt import GetoptError, getopt
from keras.models import model_from_json, load_model
from scipy.misc import imresize
import glob


'''
By Sara Bahaadini and Neda Rohani, IVPL, Northwestern University.
This function reads the trained ML classifier and pickle files of unlabelled glitches and generate the score
file in a .csv file
'''

def main(image_data, model_adr, image_size=[140, 170], verbose=False):
    """Parameters
    ----------
    image_data : `pd.DataFrame` this is a DF with
                 columns whose names are the same as
                 the image and whose row entries are
                 the b/w pixel values at some resoltion
                 determined by `make_pickle`
    model_adr : `str` path to folder containing model
    image_size : `list`, default [140, 170]
    verbose : `boolean`, default False

    Returns
    -------
    A dict with keys of workflow IDs and values list
    of golden sets associated with that workflow
    """

    dw = label_glitches(pickle_adr, model_adr, image_size, verbose)
    dwslice = dw[0][1:]
    dwslice = np.array(map(float, dwslice))

    return dw[0],np.argmax(dwslice)    

def label_glitches(image_data, model_adr, image_size=[140, 170], verbose=False):
    """Parameters
    ----------
    image_data : `pd.DataFrame` this is a DF with
                 columns whose names are the same as
                 the image and whose row entries are
                 the b/w pixel values at some resoltion
                 determined by `make_pickle`
    model_adr : `str` path to folder containing model
    image_size : `list`, default [140, 170]
    verbose : `boolean`, default False

    Returns
    -------
    A dict with keys of workflow IDs and values list
    of golden sets associated with that workflow
    """

    # the path where the trained is saved there
    model_adr += '/'

    np.random.seed(1986)  # for reproducibility

    img_rows, img_cols = image_size[0], image_size[1] 

    # load a model and weights
    if verbose:
        print ('Retrieving the trained ML classifier')
    load_folder = model_adr
    final_model = load_model(load_folder + '/multi_view_classifier.h5')

    final_model.compile(loss='categorical_crossentropy',
                        optimizer='adadelta',
                        metrics=['accuracy'])

    if verbose:
        print ('Scoring unlabelled glitches')

    # read in 4 durations
    test_set_unlabelled_x_1 = image_data.filter(regex=("0.5.png")).iloc[0].iloc[0].reshape(-1, 1, img_rows, img_cols)
    test_set_unlabelled_x_2 = image_data.filter(regex=("4.0.png")).iloc[0].iloc[0].reshape(-1, 1, img_rows, img_cols)
    test_set_unlabelled_x_3 = image_data.filter(regex=("1.0.png")).iloc[0].iloc[0].reshape(-1, 1, img_rows, img_cols)
    test_set_unlabelled_x_4 = image_data.filter(regex=("2.0.png")).iloc[0].iloc[0].reshape(-1, 1, img_rows, img_cols)

    concat_test_unlabelled = square_early_concatenate_feature(test_set_unlabelled_x_1,
                            test_set_unlabelled_x_2, test_set_unlabelled_x_3, test_set_unlabelled_x_4, [img_rows, img_cols])

    score3_unlabelled = final_model.predict_proba(concat_test_unlabelled, verbose=0)

    return score3_unlabelled, np.argmax(score3_unlabelled)


def get_feature_space(image_data, semantic_model_adr, image_size=[140, 170], verbose=False):
    """Parameters
    ----------
    image_data : `pd.DataFrame` this is a DF with
                 columns whose names are the same as
                 the image and whose row entries are
                 the b/w pixel values at some resoltion
                 determined by `make_pickle`
    semantic_model_adr : `str` path to folder containing similarity model
    image_size : `list`, default [140, 170]
    verbose : `boolean`, default False

    Returns
    -------
    A dict with keys of workflow IDs and values list
    of golden sets associated with that workflow
    """

    img_rows, img_cols = image_size[0], image_size[1]
    semantic_idx_model = load_model(semantic_model_adr + '/semantic_idx_model.h5')
    test_data = image_data.filter(regex=("1.0.png")).iloc[0].iloc[0].reshape(-1, 1, img_rows, img_cols)
    test_data = test_data.reshape([test_data.shape[0], img_rows, img_cols, 1])
    test_data = np.repeat(test_data, 3, axis=3)
    new_data2 = []
    for i in test_data:
        new_data2.append(imresize(i, (224, 224)))
        img_cols = 224
        img_rows = 224
        test_data = np.asarray(new_data2)

    return semantic_idx_model.predict([test_data])
