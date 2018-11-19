# -*- encoding: utf-8 -*-
# @Software: PyCharm
# @File    : networks.py
# @Time    : 2018/11/5 14:42
# @Author  : LU Tianle

"""
discriminator and generator networks
"""

import tensorflow as tf
import functools


class GeneratorDcgan:
    """
    discriminator and generator using the networks of DCGAN
    """

    def __init__(self, image_shape, first_conv_trans_layer_filters, conv_trans_layers, noise_dim):
        """
        :param image_shape: output image shape [channel, height, width]
        :param first_conv_trans_layer_filters: the numbers of filters in the first conv2d_transpose layer of the generator
        :param conv_trans_layers: the numbers of conv2d_transpose layers of the generator
        :param noise_dim: noise dimension
        """
        [self.channel, self.height, self.width] = image_shape
        assert (self.height / (2 ** conv_trans_layers)) % 1 == 0 and (self.width / (2 ** conv_trans_layers)) % 1 == 0, \
            'the height or width of the output images are not compatible with the conv2d_trans layers'
        assert conv_trans_layers >= 2, 'there must be more than 2 conv2d_transpose layers of the generator'
        self.conv_trans_first_layer_filters = first_conv_trans_layer_filters
        self.conv_trans_layers = conv_trans_layers
        self.noise_dim = noise_dim

    def __call__(self, batch_z, training):
        """
        create the generator with the input conv2d_transpose layer numbers and first layer filter numbers.
        the first layer is a fully connected layer with batch normalization
            that projects and reshapes the random noise followed by several conv2d_transpose layers
        every conv2d_transpose layers has kernel_size=(5, 5), strides=[2, 2] and the kernel_initializer is tf.truncated_normal_initializer(stddev=0.02)
        batch normalization and swish activation is applied to every layer except the output layer
        the output layer is also a conv2d_transpose layers with a tanh activation and no batch normalization
        layer structures:
        random noise -> project, batch_norm,swish and reshape: [batch_size, first_layer_filters, height / (2 ** layers), width / (2 ** layers)]
        ->conv2d_trans_batch_norm_swish_1: [batch_size, first_layer_filters, height / (2 ** layers), width / (2 ** layers)]
        ->conv2d_trans_batch_norm_swish_2: [batch_size, first_layer_filters // 2, height / (2 ** (layers-1)), width / (2 ** (layers-1))]
        ->...
        ->conv2d_trans_batch_norm_swish_(layers-1)
        ->conv2d_trans_tanh_layers: [batch_size, channel, height, width]
        the height or width of the output images are not compatible with the conv2d_trans layers
        the output tensor is 'generator/output:0'
        """
        # the first layer is project and reshape the random noise
        project_shape = [-1, self.conv_trans_first_layer_filters, self.height // (2 ** self.conv_trans_layers), self.width // (2 ** self.conv_trans_layers)]
        batch_z = tf.layers.dense(batch_z, units=functools.reduce(lambda x, y: abs(x) * y, project_shape), use_bias=False,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="generator/project/project")
        batch_z = tf.layers.batch_normalization(batch_z, epsilon=1e-5, name='generator/project/batch_norm', training=training)
        batch_z = tf.nn.swish(batch_z, name='generator/project/swish')
        batch_z = tf.reshape(batch_z, shape=project_shape, name='generator/reshape')
        for layer in range(self.conv_trans_layers - 1):  # conv2d transpose layers with batch normalization
            # the filters in each layer should not less than 16
            filters = self.conv_trans_first_layer_filters // (2 ** layer) if self.conv_trans_first_layer_filters / (2 ** layer) >= 16 else 16
            batch_z = tf.layers.conv2d_transpose(batch_z, filters=filters, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False,
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first",
                                                 name='generator/conv_trans_%d/conv_trans' % (layer + 1))
            batch_z = tf.layers.batch_normalization(batch_z, epsilon=1e-5, name='generator/conv_trans_%d/batch_normalization' % (layer + 1), training=training)
            batch_z = tf.nn.swish(batch_z)
        # output layer whose output shape is [batch_size, channel, height, width], no batch normalization
        batch_z = tf.layers.conv2d_transpose(batch_z, filters=self.channel, kernel_size=(5, 5), strides=(2, 2), padding='same', data_format="channels_first",
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='generator/output_layer/conv_trans')
        batch_z = tf.nn.tanh(batch_z, name='generator/output')
        return batch_z

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')


class DiscriminatorDcgan:
    def __init__(self, first_layer_filters, conv_layers):
        """
        :param image_shape: output image shape [channel, height, width]
        :param first_layer_filters:  the numbers of filters in the first conv2d layer of the discriminator
        :param conv_layers: the numbers of conv2d layers of the discriminator
        """
        assert conv_layers >= 2, 'there must be more than 1 conv2d layers of the discriminator'
        self.disc_first_layer_filters = first_layer_filters
        self.disc_conv_layers = conv_layers

    def __call__(self, batch, training):
        """
        create the discriminator with the input conv2d layer numbers and first layer filter numbers.
        each layer has first_layer_filters*(2**(layer_index-1)) filters and no pooling are used.
        every conv2d layers with kernel_size=(5, 5), strides=[2, 2] and the activation is swish
        batch normalization is applied to every layer except the input layer
        the last layer is a fully connected layer with linear activation.
        all the kernel_initializer is tf.truncated_normal_initializer(stddev=0.02)
        the output tensor is 'discriminator/output:0'
       """
        # it seems 'with tf.variable_scope('discriminator')' has no effect on the variables name of tf.layers.
        # input layer, conv2d with no batch normalization
        batch = tf.layers.conv2d(batch, filters=self.disc_first_layer_filters, kernel_size=(5, 5), strides=[2, 2],
                                 padding='same', data_format="channels_first", activation=tf.nn.swish,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='discriminator/conv_1/conv2d')
        for layer in range(2, self.disc_conv_layers + 1):  # conv2d and batch normalization from the second layer
            batch = tf.layers.conv2d(batch, filters=(2 ** (layer - 1)) * self.disc_first_layer_filters, kernel_size=(5, 5), strides=[2, 2],
                                     padding='same', data_format="channels_first", use_bias=False,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='discriminator/conv_%d/conv2d' % layer)
            batch = tf.layers.batch_normalization(batch, epsilon=1e-5, name='discriminator/conv_%d/batch_normalization' % layer, training=training)
            batch = tf.nn.swish(batch, name='discriminator/conv_%d/swish' % layer)
        # output layer, no sigmoid
        batch = tf.layers.dense(batch, units=1, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="discriminator/output/fc")
        batch = tf.layers.batch_normalization(batch, epsilon=1e-5, name='discriminator/output', training=training)
        return batch

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
