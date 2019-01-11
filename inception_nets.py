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


def __spectral_normalization(weights, iteration=1):
    """

    :param w:
    :param iteration: iterations for solving spectral norm using power iteration algorithm
    :return:
    """
    weights_shape = weights.shape.as_list()  # [h,w,in,out]
    weights = tf.reshape(weights, [-1, weights_shape[-1]])  # [out,(h*w*in)]
    u = tf.get_variable("u", [1, weights_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):  # power iteration, Usually iteration = 1 will be enough
        v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(weights)))
        u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, weights))
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, weights), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        weights_norm = tf.reshape(weights / sigma, weights_shape)
    return weights_norm


