import numpy as np
import getopt, sys, os
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rescale

'''
By Sara Bahaadini and Neda Rohani, IVPL, Northwestern University.
This function reads all glitch images with four durations from "dataset_path" and save them to four
.pkl files (one for each duration) in the "save_address"
options:
-i dataset_path, gives the adress of input image files.
-p save_address, the path where the pickle files are saved there.
-f test_flag: 1, if we are making pickles for unlabeled glitches
              0,  if we are making pickles for golden set images

note: small 'class2idx.csv' file is saved in data_path, that shows the indexing from glitch classes to numbers
'''


def main(filename, resolution=0.3, normalization_factor=255.0, verbose=False):

    np.random.seed(1986)  # for reproducibility

    image_data = io.imread(filename)
    image_data = image_data[66:532, 105:671, :]
    image_data = rgb2gray(image_data)
    image_data = rescale(image_data, resolution, mode='constant', preserve_range='True')
    dim = np.int(reduce(lambda x, y: x * y, image_data.shape))
    image_data = np.reshape(image_data, (dim))
    image_data = np.array(image_data, dtype='f')

    return image_data
