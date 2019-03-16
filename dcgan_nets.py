# -*- encoding: utf-8 -*-
# @Software: PyCharm
# @File    : dcgan_nets.py
# @Time    : 2018/11/5 14:42
# @Author  : LU Tianle

"""
discriminator and generator networks
"""

import functools
import tensorflow as tf

from layers_with_spectral_norm import Conv2D
from layers_with_spectral_norm import Dense


class Generator:
    """
    discriminator and generator using the networks of DCGAN
    """

    def __init__(self, image_shape, first_conv_trans_layer_filters, conv_trans_layers, noise_dim):
        """
        create the generator with the input conv2d_transpose layer numbers and first layer filter numbers.
        the first layer is a fully connected layer with batch normalization
            that projects and reshapes the random noise followed by several conv2d_transpose layers
        every conv2d_transpose layers has kernel_size=(3, 3) strides=[2, 2] and filters = first_conv_trans_layer_filters // (2 ** (layer - 1))
        the conv_1x1 with batch_norm layer is optional and it has half of the filters of it's corresponding conv2d_transpose
        all the kernel_initializer is tf.truncated_normal_initializer(stddev=0.02)
        batch normalization and swish activation is applied to every layer except the output layer
        the output layer(not included in the conv_trans_layers) is a conv2d_transpose layers with kernel_size=(5, 5), strides=[1, 1], tanh and no batch_norm
        :param image_shape: output image shape [channel, height, width]
        :param first_conv_trans_layer_filters: the numbers of filters in the first conv2d_transpose layer of the generator
        :param conv_trans_layers: the numbers of conv2d_transpose layers of the generator
        """
        [self.channel, self.height, self.width] = image_shape
        assert conv_trans_layers >= 2, 'there must be more than 2 conv2d_transpose layers of the generator'
        # the first layer is project and reshape the random noise
        self.noise_dim = noise_dim
        self.project_shape = [-1, 2 * first_conv_trans_layer_filters, self.height // (2 ** conv_trans_layers), self.width // (2 ** conv_trans_layers)]
        self.project = tf.layers.Dense(units=functools.reduce(lambda x, y: abs(x) * y, self.project_shape), use_bias=False,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="generator/project/project")
        self.project_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='generator/project/batch_norm')
        self.conv_trans_batch_norm_layers = []  # conv2d transpose layers with batch normalization
        for layer in range(1, conv_trans_layers):
            layer_name = 'generator/conv_trans_%d/' % layer
            filters = first_conv_trans_layer_filters // (2 ** (layer - 1))
            conv_trans = tf.layers.Conv2DTranspose(filters=filters, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False,
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), data_format="channels_first",
                                                   name=layer_name + 'conv_trans')
            batch_norm = tf.layers.BatchNormalization(axis=1, epsilon=1e-5, momentum=0.9, name=layer_name + 'conv_trans_batch_norm')
            self.conv_trans_batch_norm_layers.append((conv_trans, batch_norm, layer_name))
        # output layer whose output shape is [batch_size, channel, height, width], no batch normalization
        self.output_layer = tf.layers.Conv2DTranspose(filters=self.channel, kernel_size=(5, 5), strides=(2, 2), padding='same', data_format="channels_first",
                                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                      name='generator/conv_trans_%d/conv_trans' % conv_trans_layers)

    def __call__(self, batch_z, training, name):
        """generate images by random noise(batch_z)
        """
        batch_z = self.project(batch_z)
        batch_z = self.project_batch_norm(batch_z, training=training)
        batch_z = tf.nn.relu(batch_z, name=('generator/project/activation/' + name))
        batch_z = tf.reshape(batch_z, shape=self.project_shape, name=('generator/project/reshape/' + name))
        for (conv_trans, batch_norm, layer_name) in self.conv_trans_batch_norm_layers:
            batch_z = conv_trans(batch_z)
            batch_z = batch_norm(batch_z, training=training)
            batch_z = tf.nn.relu(batch_z, name='%sconv_5x5_activation/%s' % (layer_name, name))
            if batch_z.shape.as_list()[2:] == [6, 6]:  # MNIST
                batch_z = tf.pad(batch_z, paddings=[[0, 0], [0, 0], [0, 1], [0, 1]], mode='REFLECT', name='generator/padding')
        batch_z = self.output_layer(batch_z)
        batch_z = tf.nn.tanh(batch_z, name=('generator/output_layer/tanh/during_' + name))
        return batch_z

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')


class Discriminator:
    def __init__(self, first_layer_filters, conv_layers, spectral_norm):
        """
        create the discriminator with the input conv2d layer numbers and first layer filter numbers.
        each layer has first_layer_filters*(2**(layer_index-1)) filters and no pooling are used.
        every conv2d layers has a kernel_size=(5, 5)(the last conv2d layers is (3,3)), a strides=[2, 2] and a swish activation
        batch normalization is applied to every layer except the first conv and conv_1x1 layer (if this conv_1x1 exist)
        the last layer is a fully connected layer with linear activation and no batch normalization.
        all the kernel_initializer is tf.truncated_normal_initializer(stddev=0.02)
        :param first_layer_filters:  the numbers of filters in the first conv2d layer of the discriminator
        :param conv_layers: the numbers of conv2d layers of the discriminator
        :param spectral_norm: use Spectral Normalization
        :param use_conv_1x1: use conv_1x1 layer after each conv layer
        """
        assert conv_layers >= 2, 'there must be more than 1 conv2d layers of the discriminator'
        self.conv_batch_norm_layers = []
        for layer in range(1, conv_layers + 1):
            layer_name = 'discriminator/conv_%d/' % layer
            filters = int((2 ** (layer - 1))) * first_layer_filters
            conv = Conv2D(filters=filters, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                          use_bias=False, name=layer_name + 'conv_5x5', spectral_norm=spectral_norm, padding='same')
            if layer == 1:  # conv_1 has no batch normalization
                batch_norm = None
            else:
                batch_norm = tf.layers.BatchNormalization(axis=1, epsilon=1e-5, momentum=0.9, name=layer_name + 'conv_batch_norm')
            self.conv_batch_norm_layers.append((conv, batch_norm, layer_name))
        self.output_layer_flatten = tf.layers.Flatten(name='discriminator/output_layer/flatten', data_format='channels_first')
        self.output_layer_fc = Dense(units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=True,
                                     spectral_norm=spectral_norm, name="discriminator/output_layer/fc")

    def __call__(self, batch, training, name):
        """
         discriminate the input images, output the batch of non-normalized probabilities
        :param batch:
        :param training:
        :param name: training_generator or training_discriminator
        :return:
        """
        for (conv, batch_norm, layer_name) in self.conv_batch_norm_layers:
            batch = conv(batch)
            if batch_norm is not None:  # the first conv layer has no batch norm
                batch = batch_norm(batch, training=training)
            batch = tf.nn.leaky_relu(batch, name='%sconv_5x5_activation/%s' % (layer_name, name))
        batch = self.output_layer_flatten(batch)
        batch = self.output_layer_fc(batch)
        return batch

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
