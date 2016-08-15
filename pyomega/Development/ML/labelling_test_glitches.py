from GS_utils import load_dataset_unlabelled_glitches
import numpy as np
from keras.utils import np_utils
from functions_fusion import square_early_concatenate_feature
import sys, gzip, cPickle, os
from getopt import GetoptError, getopt
from keras.models import model_from_json
import glob


'''
By Sara Bahaadini and Neda Rohani, IVPL, Northwestern University.
This function reads the trained ML classifier and pickle files of unlabelled glitches and generate the score
file in a .csv file
options:
-p pickle_adr: the path where the pickle files of unlabelled gliches are saved there.
-m model_adr: the path where the trained model is saved there.
-s save_adr, the .csv file is saved in this address
'''

def main(pickle_adr,model_adr,save_adr,verbose):

    # Pickles of unlabelled glitches have already saved in this address
    pickle_adr += '/'

    # the path where the trained is saved there
    model_adr += '/'

    # the path where the .csv files of the results are saved there
    save_adr += '/'

    if not os.path.exists(save_adr):
        if verbose:
            print ('making... ' + save_adr)
        os.makedirs(save_adr)

    np.random.seed(1986)  # for reproducibility

    img_rows, img_cols = 47, 57
    nb_classes = 20

    # load a model and weights
    if verbose:
        print ('Retrieving the trained ML classifier')
    load_folder = model_adr
    f = gzip.open(load_folder + '/model.pklz', 'rb')
    json_string = cPickle.load(f)
    f.close()
    final_model = model_from_json(json_string)
    final_model.load_weights(load_folder + '/model_weights.h5')

    final_model.compile(loss='categorical_crossentropy',
                        optimizer='adadelta',
                        metrics=['accuracy'])

    if verbose:
        print ('Scoring unlabelled glitches')

    # reading all 4 duration pickles
    unlabelled_pickles = ['img_1.0*', 'img_2.0*', 'img_4.0*', 'img_5.0*']  # adding option to do in in alphabetical order

    # read duration 1 second

    dataset_test_unlabelled_1 = load_dataset_unlabelled_glitches(glob.glob(pickle_adr + unlabelled_pickles[0])[0],verbose)
    [test_set_unlabelled_x_1, test_set_unlabelled_y_1, test_set_unlabelled_name_1] = dataset_test_unlabelled_1
    test_set_unlabelled_x_1 = test_set_unlabelled_x_1.reshape(-1, 1, img_rows, img_cols)

    dataset_test_unlabelled_2 = load_dataset_unlabelled_glitches(glob.glob(pickle_adr + unlabelled_pickles[1])[0],verbose)
    [test_set_unlabelled_x_2, test_set_unlabelled_y_2, test_set_unlabelled_name_2] = dataset_test_unlabelled_2
    test_set_unlabelled_x_2 = test_set_unlabelled_x_2.reshape(-1, 1, img_rows, img_cols)

    dataset_test_unlabelled_3 = load_dataset_unlabelled_glitches(glob.glob(pickle_adr + unlabelled_pickles[2])[0],verbose)
    [test_set_unlabelled_x_3, test_set_unlabelled_y_3, test_set_unlabelled_name_3] = dataset_test_unlabelled_3
    test_set_unlabelled_x_3 = test_set_unlabelled_x_3.reshape(-1, 1, img_rows, img_cols)

    dataset_test_unlabelled_4 = load_dataset_unlabelled_glitches(glob.glob(pickle_adr + unlabelled_pickles[3])[0],verbose)
    [test_set_unlabelled_x_4, test_set_unlabelled_y_4, test_set_unlabelled_name_4] = dataset_test_unlabelled_4
    test_set_unlabelled_x_4 = test_set_unlabelled_x_4.reshape(-1, 1, img_rows, img_cols)

    if verbose:
        print('The number of unlabelled glitches is: ', test_set_unlabelled_x_1.shape[0])


    concat_test_unlabelled = square_early_concatenate_feature(test_set_unlabelled_x_1, \
                            test_set_unlabelled_x_2, test_set_unlabelled_x_3, test_set_unlabelled_x_4,[img_rows, img_cols])

    score3_unlabelled = final_model.predict_proba(concat_test_unlabelled, verbose=0)

    name_array_unlabelled = np.array([test_set_unlabelled_name_1.tolist()])
    name_array_unlabelled = np.transpose(name_array_unlabelled)

    dw = np.concatenate((name_array_unlabelled, score3_unlabelled), axis=1)

    return dw[0],np.argmax(score3_unlabelled[0])

if __name__ == "__main__":
   print 'Start ...'
   main(pickle_adr,model_adr,save_adr,verbose)
   print 'Done!'
