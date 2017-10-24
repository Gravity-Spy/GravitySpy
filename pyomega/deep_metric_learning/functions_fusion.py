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


def generate_shuffeled_train(number_of_shuffles, list1, list2, list3, list4):
    [train_set_x_1, train_set_y_1, train_set_name_1] = list1
    [train_set_x_2, train_set_y_2, train_set_name_2] = list2
    [train_set_x_3, train_set_y_3, train_set_name_3] = list3
    [train_set_x_4, train_set_y_4, train_set_name_4] = list4
    img_row, img_col = train_set_x_1.shape[2],train_set_x_1.shape[3]

    class_nb = max(train_set_y_1) + 1
    output = np.zeros((len(train_set_x_1)*(number_of_shuffles+1), 1, img_row * 2, img_col*2))
    output_label = np.ones((len(train_set_x_1)*(number_of_shuffles+1)))

    original_concat_train = square_early_concatenate_feature(train_set_x_1, train_set_x_2,
                                                           train_set_x_3, train_set_x_4,[img_row, img_col])

    output[0: 0 + len(original_concat_train), :, :, :] = original_concat_train
    output_label[0:0 + len(original_concat_train)] = train_set_y_1

    idx = len(original_concat_train)
    #idx = 0

    for it in range(number_of_shuffles):
        for c in range(class_nb):
            dur1 = np.where(train_set_y_1 == c)
            dur2 = np.where(train_set_y_2 == c)
            dur3 = np.where(train_set_y_3 == c)
            dur4 = np.where(train_set_y_4 == c)

            d1 = range(len(dur1[0]))
            d2 = range(len(dur2[0]))
            d3 = range(len(dur3[0]))
            d4 = range(len(dur4[0]))

            np.random.shuffle(d1)
            np.random.shuffle(d2)
            np.random.shuffle(d3)
            np.random.shuffle(d4)

            shuf_dur1 = dur1[0][d1]
            shuf_dur2 = dur2[0][d2]
            shuf_dur3 = dur3[0][d3]
            shuf_dur4 = dur4[0][d4]

            temp_concat_class_c = square_early_concatenate_feature(train_set_x_1[shuf_dur1], train_set_x_2[shuf_dur2],
                                             train_set_x_3[shuf_dur3], train_set_x_4[shuf_dur4], [img_row, img_col])

            output[idx:idx+len(temp_concat_class_c), :, :, :] = temp_concat_class_c
            output_label[idx:idx + len(temp_concat_class_c)] = train_set_y_1[shuf_dur1]
            idx = idx + len(temp_concat_class_c)
    return output, output_label
