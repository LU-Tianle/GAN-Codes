# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : inception_module.py 
# @Time    : 2019/1/6 15:27
# @Author  : LU Tianle

"""
Inception Module and Inception-Trans Module
"""
import tensorflow as tf


class InceptionModule():
    def __init__(self, first_layer_filters):
        pass

    def __call__(self, batch, training, batch_norm=True):
        pass


class InceptionTransModule():
    def __init__(self, first_layer_filters):
        pass

    def __call__(self, batch, training, batch_norm=True):
        pass

# def conv2d(x, filters, kernel_size, name, strides=(1, 1), padding='valid', spectral_norm=False):
#     """
#
#     :param x:
#     :param channel:
#     :param k_h:
#     :param k_w:
#     :param d_h:
#     :param d_w:
#     :param stddev:
#     :param name:
#     :return:
#     """
#     with tf.variable_scope(name):
#         w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], filters], initializer=tf.truncated_normal_initializer(stddev=stddev))
#         if spectral_norm:
#             w_sn = __spectral_norm(w, iteration=3)
#         conv = tf.nn.conv2d(x, filter=w_sn, strides=[1, d_h, d_w, 1], padding='VALID')
#         biases = tf.get_variable('biases', [channel], initializer=tf.constant_initializer(0.0))
#         conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
#         return conv
