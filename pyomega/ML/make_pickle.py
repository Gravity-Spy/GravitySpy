
from GS_utils import my_read_image, my_save_dataset_test, my_save_dataset, listdir_nohidden
import numpy as np
import getopt, sys, os


'''
By Sara Bahaadini and Neda Rohani, IVPL, Northwestern University.
This function reads all glitch images with four durations from "dataset_path" and save them to four
.pkl files (one for each duration) in the "save_address"
options:
-i dataset_path, gives the adress of input image files.
-p save_address, the path where the pickle files are saved there.
-f test_flag: 1, if we are making pickles for unlabeled glitches
              0,  if we are making pickles for golden set images

note: small 'class2idx.csv' file is saved in current directory, that shows the indexing from glitch classes to numbers
'''

def main(dataset_path,save_address,test_flag):

    # Default input dataset path
    dataset_path += '/'

    # default address to save pickle files
    save_address += '/'

    # if input dataset is the collection of unlabelled glitches it is "1",
    # if it is golden set set, it is "0".

    print 'input dataset path', dataset_path
    print 'save adress', save_address
    print 'test flag', test_flag
    if not os.path.exists(save_address):
        print ('making... ' + save_address)
        os.makedirs(save_address)

    # these lines read all images with duration 1.0, 0.5, 2.0 and 4 seconds

    imgs_1, label1, name1 = my_read_image(dataset_path, 'spectrogram_1.0.png')
    imgs_1_fl = np.array(imgs_1, dtype='f')

    "shuffling indexes for splitting data into training, validation, and test"
    data_size = imgs_1_fl.shape[0]
    new_ind = np.random.permutation(range(data_size))

    imgs_5, label5, name5 = my_read_image(dataset_path, 'spectrogram_0.5.png')
    imgs_5_fl = np.array(imgs_5, dtype='f')

    imgs_2, label2, name2 = my_read_image(dataset_path, 'spectrogram_2.0.png')
    imgs_2_fl = np.array(imgs_2, dtype='f')

    imgs_4, label4, name4 = my_read_image(dataset_path, 'spectrogram_4.0.png')
    imgs_4_fl = np.array(imgs_4, dtype='f')

    # check if all four durations for each image exist
    er1 = set(name1) ^ set(name2)
    er2 = set(name2) ^ set(name4)
    er3 = set(name5) ^ set(name4)

    if (len(er1)):
        print ('er1=', er1)
        sys.exit('Error!! these images do not have all 4 durations')
    if (len(er2)) :
        print ('er2=', er2)
        sys.exit('Error!! these images do not have all 4 durations')
    if  (len(er3)):
        print ('er3=', er3)
        sys.exit('Error!! these images do not have all 4 durations')

    # writing into pickle files
    if test_flag:
        # for unlabelled glitches
        my_save_dataset_test(save_address, imgs_1_fl, label1, name1, '_1.0_')
        my_save_dataset_test(save_address, imgs_4_fl, label4, name4, '_4.0_')
        my_save_dataset_test(save_address, imgs_2_fl, label2, name2, '_2.0_')
        my_save_dataset_test(save_address, imgs_5_fl, label5, name5, '_5.0_')
        # for golden set
    else:
        my_save_dataset(save_address, new_ind, imgs_1_fl, label1, name1, '_1.0_')
        my_save_dataset(save_address, new_ind, imgs_4_fl, label4, name4, '_4.0_')
        my_save_dataset(save_address, new_ind, imgs_2_fl, label2, name2, '_2.0_')
        my_save_dataset(save_address, new_ind, imgs_5_fl, label5, name5, '_5.0_')

if __name__ == "__main__":

    print "Start making pickles!"
    main(dataset_path,save_address,test_flag)
    print('Done!')
