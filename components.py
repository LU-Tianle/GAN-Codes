# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : components.py
# @Time    : 2018/11/15 10:07
# @Author  : LU Tianle

"""
"""

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np


def create_folder(path, continue_training=False):
    """
    create a folder if it's not exist
    :param path: folder path,
    :param continue_training: if False, all folders and files in the path will be deleted recursively
    """
    if os.path.exists(path):
        if not continue_training:
            shutil.rmtree(path)  # Recursively delete a directory tree
            os.makedirs(path)
    else:
        os.makedirs(path)


def save_parameters(first_conv_trans_layer_filters, conv_trans_layers, first_layer_filters, conv_layers,
                    dataset, batch_size, noise_dim, training_algorithm, save_path):
    """
    save networks and training parameters
    """
    file = open(save_path + os.path.sep + "parameter.txt", mode='w')
    file.write('FIRST_CONV_TRANS_LAYER_FILTERS ' + str(first_conv_trans_layer_filters) + '\n')
    file.write('CONV_TRANS_LAYERS ' + str(conv_trans_layers) + '\n')
    file.write('\n')
    file.write('FIRST_LAYER_FILTERS ' + str(first_layer_filters) + '\n')
    file.write('CONV_LAYERS ' + str(conv_layers) + '\n')
    file.write('\n')
    file.write('DATASET ' + dataset + '\n')
    file.write('BATCH_SIZE ' + str(batch_size) + '\n')
    file.write('NOISE_DIM ' + str(noise_dim) + '\n')
    file.write('TRAINING_ALGORITHM ' + training_algorithm + '\n')
    file.write('\n')
    file.write('CHECK_POINTS_PATH ' + save_path + os.path.sep + 'check_points' + '\n')
    file.write('GENERATED_IMAGE_DURING_PATH ' + save_path + os.path.sep + 'images_during_training' + '\n')
    file.write('\n')
    file.close()


def __generate_and_save_images(output_path, predictions, images_per_row, epoch, index=0):
    fig = plt.figure(figsize=(images_per_row, images_per_row))
    fig.suptitle('epoch_{:04d}.png'.format(epoch))
    for i in range(images_per_row ** 2):
        plt.subplot(images_per_row, images_per_row, i + 1)
        if predictions.shape[1] == 1:
            plt.imshow(predictions[i][0] * 127.5 + 127.5, cmap='gray')
        else:
            image = np.around(predictions[i] * 127.5 + 127.5).astype(int)
            image = np.transpose(image, [2, 1, 0])
            plt.imshow(image)
            plt.axis('off')
    plt.savefig(output_path + os.path.sep + 'epoch_{:04d}_{}.png'.format(epoch, index))
    plt.close(fig)
