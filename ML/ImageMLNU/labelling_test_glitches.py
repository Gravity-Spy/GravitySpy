from GS_utils import load_dataset_unlabelled_glitches
import numpy as np
from keras.utils import np_utils
from functions_fusion import square_early_concatenate_feature
import sys, gzip, cPickle, os
from getopt import GetoptError, getopt
from keras.models import model_from_json
import optparse


'''
By Sara Bahaadini and Neda Rohani, IVPL, Northwestern University.
This function reads the trained ML classifier and pickle files of unlabelled glitches and generate the score
file in a .csv file
options:
-p pickle_adr: the path where the pickle files of unlabelled gliches are saved there.
-m model_adr: the path where the trained model is saved there.
-s save_adr, the .csv file is saved in this address
'''

def parse_commandline():
    """Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--pickle-address",default='./pickeled/',help="path to unlabelled images")
    parser.add_option("--save-address",default='./dataout/',help="path to save the pickled images")
    parser.add_option("--model-address",default='./model/',help="path to save the pickled images")
    opts, args = parser.parse_args()


    return opts


def main():

    opts = parse_commandline()

    # Pickles of unlabelled glitches have already saved in this address
    pickle_adr = opts.pickle_address
    pickle_adr += '/'

    # the path where the trained is saved there
    model_adr = opts.model_address
    model_adr += '/'

    # the path where the .csv files of the results are saved there
    save_adr = opts.save_address
    save_adr += '/'

    if not os.path.exists(save_adr):
                print ('making... ' + save_adr)
                os.makedirs(save_adr)

    np.random.seed(1986)  # for reproducibility

    img_rows, img_cols = 47, 57
    nb_classes = 20

    # load a model and weights
    print ('Retrieving the trained ML classifier')
    load_folder = model_adr
    f = gzip.open(load_folder + '/model.pklz', 'rb')
    json_string = cPickle.load(f)
    f.close()
    final_model = model_from_json(json_string)
    final_model.load_weights(load_folder + '/model_weights.h5')

    final_model.compile(loss='categorical_crossentropy',
                        optimizer='adadelta',
                        # optimizer=model_optimizer,
                        metrics=['accuracy'])

    print ('Scoring unlabelled glitches')

    # reading all 4 duration pickles
    unlabelled_pickles = os.listdir(pickle_adr)  # adding option to do in in alphabetical order

    # read duration 1 second
    #dataset_test_unlabelled_1 = load_dataset_unlabelled_glitches(pickle_adr + 'img_1.0_class2_norm.pkl.gz')
    dataset_test_unlabelled_1 = load_dataset_unlabelled_glitches(pickle_adr + unlabelled_pickles[0])
    [test_set_unlabelled_x_1, test_set_unlabelled_y_1, test_set_unlabelled_name_1] = dataset_test_unlabelled_1
    test_set_unlabelled_x_1 = test_set_unlabelled_x_1.reshape(-1, 1, img_rows, img_cols)

    dataset_test_unlabelled_2 = load_dataset_unlabelled_glitches(pickle_adr + unlabelled_pickles[1])
    [test_set_unlabelled_x_2, test_set_unlabelled_y_2, test_set_unlabelled_name_2] = dataset_test_unlabelled_2
    test_set_unlabelled_x_2 = test_set_unlabelled_x_2.reshape(-1, 1, img_rows, img_cols)

    dataset_test_unlabelled_3 = load_dataset_unlabelled_glitches(pickle_adr + unlabelled_pickles[2])
    [test_set_unlabelled_x_3, test_set_unlabelled_y_3, test_set_unlabelled_name_3] = dataset_test_unlabelled_3
    test_set_unlabelled_x_3 = test_set_unlabelled_x_3.reshape(-1, 1, img_rows, img_cols)

    dataset_test_unlabelled_4 = load_dataset_unlabelled_glitches(pickle_adr + unlabelled_pickles[3])
    [test_set_unlabelled_x_4, test_set_unlabelled_y_4, test_set_unlabelled_name_4] = dataset_test_unlabelled_4
    test_set_unlabelled_x_4 = test_set_unlabelled_x_4.reshape(-1, 1, img_rows, img_cols)

    print('The number of unlabelled glitches is: ', test_set_unlabelled_x_1.shape[0])


    concat_test_unlabelled = square_early_concatenate_feature(test_set_unlabelled_x_1, \
                            test_set_unlabelled_x_2, test_set_unlabelled_x_3, test_set_unlabelled_x_4,[img_rows, img_cols])

    score2_unlabelled = final_model.predict_classes(concat_test_unlabelled, verbose=0)
    score2_unlabelled_array = np.array([score2_unlabelled.tolist()])
    score2_unlabelled_array = np.transpose(score2_unlabelled_array)

    score3_unlabelled = final_model.predict_proba(concat_test_unlabelled, verbose=0)

    name_array_unlabelled = np.array([test_set_unlabelled_name_1.tolist()])
    name_array_unlabelled = np.transpose(name_array_unlabelled)

    dw = np.concatenate((name_array_unlabelled, score2_unlabelled_array, score3_unlabelled), axis=1)
    np.savetxt(save_adr + '/scores.csv', dw, delimiter=',', fmt='%s')

if __name__ == "__main__":
   print 'Start ...'
   main()
   print 'Done!'
