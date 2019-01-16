# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : components.py
# @Time    : 2018/11/15 10:07
# @Author  : LU Tianle

"""
"""

import os
import shutil


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
                    discriminator_training_loop, dataset, batch_size, noise_dim, training_algorithm, save_path):
    """
    save networks and training parameters of Dcgan
    """
    file = open(save_path + os.path.sep + "parameters.txt", mode='w')
    file.write('Networks: Dcgan\n\n')
    file.write('FIRST_CONV_TRANS_LAYER_FILTERS ' + str(first_conv_trans_layer_filters) + '\n')
    file.write('CONV_TRANS_LAYERS ' + str(conv_trans_layers) + '\n\n')
    file.write('FIRST_LAYER_FILTERS ' + str(first_layer_filters) + '\n')
    file.write('CONV_LAYERS ' + str(conv_layers) + '\n\n')
    file.write('DATASET ' + dataset + '\n')
    file.write('BATCH_SIZE ' + str(batch_size) + '\n')
    file.write('NOISE_DIM ' + str(noise_dim) + '\n')
    file.write('TRAINING_ALGORITHM ' + training_algorithm + '\n')
    file.write('DISCRIMINATOR_TRAINING_LOOP ' + str(discriminator_training_loop) + '\n\n')
    file.write('CHECK_POINTS_PATH ' + save_path + os.path.sep + 'check_points' + '\n')
    file.write('GENERATED_IMAGE_DURING_PATH ' + save_path + os.path.sep + 'images_during_training' + '\n\n')
    file.close()
