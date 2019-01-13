# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : inception_nets.py 
# @Time    : 2019/1/5 14:31
# @Author  : LU Tianle

"""
"""
import tensorflow as tf


class Generator:
    def __init__(self, first_layer_filters):
        pass

    def __call__(self, batch, training, batch_norm=True):
        pass

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')


class Discriminator:
    def __init__(self, first_layer_filters):
        pass

    def __call__(self, batch, training, batch_norm=True):
        pass

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')




