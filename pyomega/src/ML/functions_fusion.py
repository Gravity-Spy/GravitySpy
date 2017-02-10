import numpy as np


def early_concatenate_feature(image_set1, image_set2, image_size):
    img_rows = image_size[0]
    img_cols = image_size[1]

    assert len(image_set1) == len(image_set2)
    concat_images_set = np.zeros((len(image_set1), 1, img_rows * 2, img_cols))
    for i in range(0, len(image_set1)):
        concat_images_set[i, :, :, :] = np.append(image_set1[i, :, :, :], image_set2[i, :, :, :], axis=1)

    return concat_images_set


def square_early_concatenate_feature(image_set1, image_set2,image_set3, image_set4, image_size):
    img_rows = image_size[0]
    img_cols = image_size[1]

    assert len(image_set1) == len(image_set2)
    concat_images_set = np.zeros((len(image_set1), 1, img_rows * 2, img_cols))
    for i in range(0, len(image_set1)):
        concat_images_set[i, :, :, :] = np.append(image_set1[i, :, :, :], image_set2[i, :, :, :], axis=1)

    #print 'concat1', concat_images_set.shape
    assert len(image_set3) == len(image_set4)
    concat_images_set2 = np.zeros((len(image_set3), 1, img_rows * 2, img_cols))
    for i in range(0, len(image_set3)):
        concat_images_set2[i, :, :, :] = np.append(image_set3[i, :, :, :], image_set4[i, :, :, :], axis=1)

    #print 'concat1', concat_images_set2.shape
    out = np.append(concat_images_set,concat_images_set2, axis=3)
    #print 'out shape', out.shape
    #return concat_images_set
    return out



def volumize_concatenate_feature(image_set_list, image_size):

    #list of input [image_set1, image_set2, ...]
    img_rows = image_size[0]
    img_cols = image_size[1]
    number_of_channels = len(image_set_list)
    number_of_images = len(image_set_list[0])
    concat_images_set = np.zeros((number_of_images, number_of_channels, img_rows, img_cols))

    for i in range(0, number_of_images):
        tmp_image = np.zeros((number_of_channels, img_rows, img_cols))
        for chanel_idx in range(0, number_of_channels):
            image_set = image_set_list[chanel_idx]
            tmp_image[chanel_idx, :, :] = image_set[i, :, :, :]

        concat_images_set[i, :, :, :] = tmp_image

    return concat_images_set
