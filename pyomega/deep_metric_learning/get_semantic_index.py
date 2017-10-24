from GS_utils import my_load_dataset
import sys, getopt
import gzip, os
from keras.models import load_model
from GS_utils import load_dataset_unlabelled_glitches
import numpy as np
from keras.utils import np_utils
from functions_fusion import square_early_concatenate_feature
import sys, gzip, pickle, os
from getopt import GetoptError, getopt
from keras.models import Model
import optparse
from scipy.misc import imresize

'''
By Sara Bahaadini.
This function loads the trained semantic index model and reads the input pickle files 
and returns the corresponding semantic indexes in a .csv file
'''

def parse_commandline():
    """Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--pickle-address", default='./pickeled/', help="path to unlabelled images")
    parser.add_option("--save-address", default='./save_adr/',
                      help="saving the semantic index of the input files there")
    parser.add_option("--model-address", default='./model/', help="path to load the trained models from")
    opts, args = parser.parse_args()
    return opts


def main(pickle_address='./pickled/', model_address='./model/'):
    multi_view = 0
    img_rows, img_cols = 47, 57
    transfer_learning = 1


    # Pickles of unlabelled glitches have already saved in this address (made by make_pickle.py)
    pickle_adr = pickle_address
    pickle_adr += '/'

    # the path where the trained is saved there
    semantic_model_adr = model_address
    semantic_model_adr += '/'

    print('Retrieving the trained semantic index model ...')
    semantic_idx_model = load_model(semantic_model_adr + '/semantic_idx_model.h5')
    files = os.listdir(pickle_adr)
    for f in files:
        if 'img_5.0' in f:
            ad1 = pickle_adr + f
        elif 'img_4.0' in f:
            ad2 = pickle_adr + f
        elif 'img_1.0' in f:
            ad3 = pickle_adr + f
        elif 'img_2.0' in f:
            ad4 = pickle_adr + f

    # reading all 4 duration pickles
    print('Reading the data ...')
    if multi_view:
        dataset_test_unlabelled_1 = load_dataset_unlabelled_glitches(ad1)
        [test_set_unlabelled_x_1, y, test_set_unlabelled_name_1] = dataset_test_unlabelled_1
        test_set_unlabelled_x_1 = test_set_unlabelled_x_1.reshape(-1, 1, img_rows, img_cols)

        dataset_test_unlabelled_2 = load_dataset_unlabelled_glitches(ad2)
        test_set_unlabelled_x_2, _, _ = dataset_test_unlabelled_2
        test_set_unlabelled_x_2 = test_set_unlabelled_x_2.reshape(-1, 1, img_rows, img_cols)

        dataset_test_unlabelled_3 = load_dataset_unlabelled_glitches(ad3)
        [test_set_unlabelled_x_3, _, _ ] = dataset_test_unlabelled_3
        test_set_unlabelled_x_3 = test_set_unlabelled_x_3.reshape(-1, 1, img_rows, img_cols)

        dataset_test_unlabelled_4 = load_dataset_unlabelled_glitches(ad4)
        [test_set_unlabelled_x_4, _, _ ] = dataset_test_unlabelled_4
        test_set_unlabelled_x_4 = test_set_unlabelled_x_4.reshape(-1, 1, img_rows, img_cols)

        concat_test = square_early_concatenate_feature(test_set_unlabelled_x_1, test_set_unlabelled_x_2
                                                       , test_set_unlabelled_x_3, test_set_unlabelled_x_4,
                                                        [img_rows, img_cols])
        test_data = concat_test
    else:
        dataset_test_unlabelled_1 = load_dataset_unlabelled_glitches(ad1)
        [test_set_unlabelled_x_1, y, test_set_unlabelled_name_1] = dataset_test_unlabelled_1
        test_set_unlabelled_x_1 = test_set_unlabelled_x_1.reshape(-1, 1, img_rows, img_cols)
        test_data = test_set_unlabelled_x_1


    test_data = test_data.astype('float32')
    if transfer_learning:
        test_data = test_data.reshape([test_data.shape[0], img_rows, img_cols, 1])
        test_data = np.repeat(test_data, 3, axis=3)
        new_data2 = []
        for i in test_data:
            new_data2.append(imresize(i, (224, 224)))
            img_cols = 224
            img_rows = 224
            test_data = np.asarray(new_data2)

    else:
        test_data = np.reshape(test_data, (test_data.shape[0], -1))

    semantic_index_features = semantic_idx_model.predict([test_data])
    data_name = test_set_unlabelled_name_1.reshape(test_set_unlabelled_name_1.shape[0], 1)
    labels = y.reshape(y.shape[0],1)
    written_data = np.append(data_name, labels, axis=1)
    written_data = np.append(written_data, semantic_index_features, axis=1)
    import pdb
    pdb.set_trace()
    np.savetxt(save_adr + '/features2.csv', written_data, delimiter=',', fmt='%s')


if __name__ == "__main__":
    print('Start ...')
    main()
    print('Done!')
