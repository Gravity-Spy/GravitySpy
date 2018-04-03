

import numpy as np
import os
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rescale
from functools import reduce
import datetime
import optparse
from GS_utils import my_save_dataset_test, my_read_image2, my_save_dataset_stratified, pre_process_img
import matplotlib.pyplot as plt


'''
This function reads all glitch images with four durations from "dataset_path" and save them to four
.pkl files (one for each duration) in the "save_address"
options:
--dataset-path, gives the adress of input image files.
--save-address, the path where the pickle files are saved there.
--test-flag: 1, if we are making pickles for unlabeled glitches
              0,  if we are making pickles for golden set images
--simi-flag: similarity measure flag if test flag is one, ignore this flag, otherwise 
            it is 1, if we are making pickles for similairty measure (ignores none of the above Class)
            0, for all the other cases.
'''


def parse_commandline():
    """Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--dataset-path", default='./data/', help="path to glitch images")
    parser.add_option("--save-address", default='./pickle_adr/', help="path to save the pickled images there")
    parser.add_option("--test-flag", type=int, default=0, help="test flag to show the pickles are for unlabeled glitches")
    parser.add_option("--simi-flag", type=int, default=0, help="similarity measure flag")
    opts, args = parser.parse_args()
    
    return opts


def make_view_rgb(file, resolution, rescale_flag):
    file_1 = pre_process_img(file[:, :, 0], resolution, rescale_flag)
    file_2 = pre_process_img(file[:, :, 1], resolution, rescale_flag)
    file_3 = pre_process_img(file[:, :, 2], resolution, rescale_flag)
    return [file_1, file_2, file_3]

def main():
    np.random.seed(1986)  # for reproducibility

    rescale_flag = 1  # if 1, the resolution is the down sampling factor, else it is the target size
    resolution = 0.3
    rgb_flag = 0

    opts = parse_commandline()
    dataset_path = opts.dataset_path
    dataset_path += '/'
    dataset_path = '/media/sara/DATA/GravitySpy/April_2_2018/trainingsetv1d1/'

    save_address = opts.save_address
    save_address = '//media/sara/DATA/GravitySpy/April_3_2018/pickles/'
    today = datetime.date.today()
    save_address = save_address + 'pickle_' + str(today) + '_res' + str(resolution) + '/'

    # if input dataset is the collection of unlabelled glitches it is "1",
    # if it is golden set set, it is "0".
    test_flag = opts.test_flag
    # flag to specify whether to exclude None of the above class in the case of golden set

    simi_flag = opts.simi_flag
    if test_flag:
        simi_flag = 0

    print('Input dataset path', dataset_path)
    print('Save address', save_address)
    print('test flag', test_flag)

    if not os.path.exists(save_address):
        print('Making... ' + save_address)
        os.makedirs(save_address)

    # these lines read all images with duration 1.0, 0.5, 2.0 and 4 seconds
    imgs_1, label1, name1, classes = my_read_image2(save_address, dataset_path, 'spectrogram_1.0.png',
                                                    resolution, simi_flag, rgb_flag, rescale_flag)
    imgs_1_fl = np.array(imgs_1, dtype='f')

    imgs_2 = []
    label2 = []
    name2 = []

    imgs_4 = []
    label4 = []
    name4 = []

    imgs_5 = []
    label5 = []
    name5 = []

    for idx, fi in enumerate(name1):
            path5 = dataset_path + '/' + classes[label1[idx]] + '/' + fi + '_spectrogram_0.5.png'
            path2 = dataset_path + '/' + classes[label1[idx]] + '/' + fi + '_spectrogram_2.0.png'
            path4 = dataset_path + '/' + classes[label1[idx]] + '/' + fi + '_spectrogram_4.0.png'

            ''' path5 = dataset_path + '/' + classes[label1[idx]] + '/' + fi + '_spectrogram_2.0.png'
            path2 = dataset_path + '/' + classes[label1[idx]] + '/' + fi + '_spectrogram_2.0.png'
            path4 = dataset_path + '/' + classes[label1[idx]] + '/' + fi + '_spectrogram_2.0.png'''

            file5 = io.imread(path5)
            file2 = io.imread(path2)
            file4 = io.imread(path4)
            #io.imshow(file5)

            file5 = file5[66:532, 105:671, :]
            file2 = file2[66:532, 105:671, :]
            file4 = file4[66:532, 105:671, :]

           # plt.show(file5)
            if not rgb_flag:
                imgs_5.append(pre_process_img(rgb2gray(file5), resolution, rescale_flag))
                imgs_2.append(pre_process_img(rgb2gray(file2), resolution, rescale_flag))
                imgs_4.append(pre_process_img(rgb2gray(file4), resolution, rescale_flag))
            else:
                imgs_5.append(make_view_rgb(file5, resolution, rescale_flag))
                imgs_2.append(make_view_rgb(file2, resolution, rescale_flag))
                imgs_4.append(make_view_rgb(file4, resolution, rescale_flag))


    name2 = name1
    name4 = name1
    name5 = name1

    label2 = label1
    label4 = label1
    label5 = label1

    imgs_5_fl = np.array(imgs_5, dtype='f')
    imgs_2_fl = np.array(imgs_2, dtype='f')
    imgs_4_fl = np.array(imgs_4, dtype='f')

    if test_flag:
        # for unlabelled glitches
        my_save_dataset_test(save_address, imgs_1_fl, label1, name1, '_1.0_', rgb_flag)
        my_save_dataset_test(save_address, imgs_4_fl, label4, name4, '_4.0_', rgb_flag)
        my_save_dataset_test(save_address, imgs_2_fl, label2, name2, '_2.0_', rgb_flag)
        my_save_dataset_test(save_address, imgs_5_fl, label5, name5, '_5.0_', rgb_flag)

    else:
        my_save_dataset_stratified(save_address, imgs_1_fl, label1, name1, '_1.0_', rgb_flag)
        my_save_dataset_stratified(save_address, imgs_4_fl, label4, name4, '_4.0_', rgb_flag)
        my_save_dataset_stratified(save_address, imgs_2_fl, label2, name2, '_2.0_', rgb_flag)
        my_save_dataset_stratified(save_address, imgs_5_fl, label5, name5, '_5.0_', rgb_flag)


if __name__ == "__main__":
    print('Start making pickles!')
    main()
    print('Done!')
