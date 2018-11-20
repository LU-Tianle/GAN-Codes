# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : utils.py 
# @Time    : 2018/11/15 10:07
# @Author  : LU Tianle

"""
"""

import os


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
