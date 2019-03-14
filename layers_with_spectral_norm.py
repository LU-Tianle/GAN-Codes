# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : layers_with_spectral_norm.py
# @Time    : 2019/1/15 19:08
# @Author  : LU Tianle

"""
layers of fully connected and conv2d with Spectral Normalization
"""
import tensorflow as tf


class Dense:
    def __init__(self, units, kernel_initializer, use_bias=False, bias_initializer=None, activation=None, spectral_norm=False, name=None):
        """
        fully connected with Spectral Normalization layer
        :param units:
        :param kernel_initializer:
        :param use_bias:
        :param bias_initializer:
        :param activation:
        :param spectral_norm:
        :param name:
        """
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.spectral_norm = spectral_norm
        self.name = name

    def __call__(self, batch):
        batch_shape = batch.get_shape().as_list()
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable('kernel', shape=[batch_shape[1], self.units], initializer=self.kernel_initializer, dtype=tf.float32)
            if self.use_bias:
                bias = tf.get_variable('bias', shape=[self.units], initializer=self.bias_initializer, dtype=tf.float32)
            else:
                bias = None
        if self.spectral_norm:
            weights = spectral_normalization(weights, iteration=1, name=self.name)
        batch = tf.matmul(batch, weights, name=self.name + '/matmul')
        if self.use_bias:
            batch = tf.add(batch, bias, name=self.name + '/add_bias')
        if self.activation is not None:
            batch = self.activation(batch, name=self.name + '/activation')
        return batch


class Conv2D:
    def __init__(self, filters, kernel_size, kernel_initializer,
                 strides=(1, 1), padding='same', use_bias=False, bias_initializer=None, activation=None,
                 spectral_norm=False, data_format="channels_first", name=None):
        """
        conv2d with Spectral Normalization layer
        :param filters:
        :param kernel_size:
        :param kernel_initializer:
        :param strides:
        :param padding:
        :param use_bias:
        :param bias_initializer:
        :param activation:
        :param spectral_norm:
        :param data_format: 'channels_first' or 'channels_last'
        :param name: layer name
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.strides = strides
        self.padding = str.upper(padding)
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.spectral_norm = spectral_norm
        self.data_format = 'NHWC' if data_format == "channels_last" else 'NCHW'
        self.name = name

    def __call__(self, batch):
        batch_shape = batch.get_shape().as_list()
        if self.data_format == 'NCHW':
            kernel_shape = self.kernel_size + [batch_shape[1], self.filters]
            strides = [1, 1] + list(self.strides)
        else:
            kernel_shape = self.kernel_size + [batch_shape[3], self.filters]
            strides = [1] + self.strides + [1]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable('kernel', shape=kernel_shape, initializer=self.kernel_initializer, dtype=tf.float32)
            if self.use_bias:
                bias = tf.get_variable('bias', shape=[self.filters], initializer=self.bias_initializer, dtype=tf.float32)
            else:
                bias = None
        if self.spectral_norm:
            kernel = spectral_normalization(kernel, iteration=1, name=self.name)
        conv2d = tf.nn.conv2d(batch, filter=kernel, strides=strides, padding=self.padding, data_format=self.data_format, name=self.name + '/conv')
        if self.use_bias:
            conv2d = tf.nn.bias_add(conv2d, bias, data_format=self.data_format, name=self.name + '/add_bias')
        if self.activation is not None:
            conv2d = self.activation(conv2d, name=self.name + '/activation')
        return conv2d


def spectral_normalization(weights, name, iteration=1):
    """
    spectral normalization
    :param weights:
    :param name:
    :param iteration: iterations for solving spectral norm using power iteration algorithm
    :return:
    """
    weights_shape = weights.shape.as_list()
    with tf.name_scope(name + '/spectral_normalization'):
        weights = tf.reshape(weights, [-1, weights_shape[-1]])
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            u = tf.get_variable("u", [1, weights_shape[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02), trainable=False)
        with tf.name_scope('power_iteration'):
            u_hat = u
            v_hat = None
            for i in range(iteration):  # power iteration, Usually iteration = 1 will be enough
                v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(weights)))
                u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, weights))
            u_hat = tf.stop_gradient(u_hat)
            v_hat = tf.stop_gradient(v_hat)
            sigma = tf.matmul(tf.matmul(v_hat, weights), tf.transpose(u_hat), name='sigma')
        with tf.control_dependencies([u.assign(u_hat)]):
            weights_norm = tf.reshape(weights / sigma, weights_shape)
    return weights_norm
