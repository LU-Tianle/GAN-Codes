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
        :param image_shape: output image shape [channel, height, width]
        :param first_conv_trans_layer_filters: the numbers of filters in the first conv2d_transpose layer of the generator
        :param conv_trans_layers: the numbers of conv2d_transpose layers of the generator
        """
        [self.channel, self.height, self.width] = image_shape
        assert (self.height / (2 ** conv_trans_layers)) % 1 == 0 and (self.width / (2 ** conv_trans_layers)) % 1 == 0, \
            'the height or width of the output images are not compatible with the conv2d_trans layers'
        assert conv_trans_layers >= 2, 'there must be more than 2 conv2d_transpose layers of the generator'
        # the first layer is project and reshape the random noise
        self.noise_dim = noise_dim
        self.project_shape = [-1, first_conv_trans_layer_filters, self.height // (2 ** conv_trans_layers), self.width // (2 ** conv_trans_layers)]
        self.project = tf.layers.Dense(units=functools.reduce(lambda x, y: abs(x) * y, self.project_shape), use_bias=False,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="generator/project/project")
        self.project_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, name='generator/project/batch_norm')
        self.conv_trans_batch_norm_layers = []  # conv2d transpose layers with batch normalization
        for layer in range(conv_trans_layers - 1):
            # the filters in each layer should not less than 8
            filters = first_conv_trans_layer_filters // (2 ** layer) if first_conv_trans_layer_filters / (2 ** layer) >= 8 else 8
            conv_trans = tf.layers.Conv2DTranspose(filters=filters, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False,
                                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first",
                                                   name='generator/conv_trans_%d/conv_trans' % (layer + 1))
            batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, name='generator/conv_trans_%d/batch_normalization' % (layer + 1))
            name = 'generator/conv_trans_%d/swish' % (layer + 1)
            self.conv_trans_batch_norm_layers.append((conv_trans, batch_norm, name))
        # output layer whose output shape is [batch_size, channel, height, width], no batch normalization
        self.output_layer = tf.layers.Conv2DTranspose(filters=self.channel, kernel_size=(5, 5), strides=(2, 2), padding='same', data_format="channels_first",
                                                      kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                                      name='generator/conv_trans_%d/conv_trans' % conv_trans_layers)

    def __call__(self, batch_z, training, name):
        """
        generate images by random noise(batch_z)
        """
        batch_z = self.project(batch_z)
        batch_z = self.project_batch_norm(batch_z, training=training)
        batch_z = tf.nn.swish(batch_z, name='generator/project/swish')
        batch_z = tf.reshape(batch_z, shape=self.project_shape, name='generator/reshape')
        for (conv_trans, batch_norm, layer_name) in self.conv_trans_batch_norm_layers:
            batch_z = conv_trans(batch_z)
            batch_z = batch_norm(batch_z, training=training)
            batch_z = tf.nn.swish(batch_z, name=layer_name)
        batch_z = self.output_layer(batch_z)
        batch_z = tf.nn.tanh(batch_z, name=name)
        return batch_z

    def generate(self):
        """
        used for generating images, generate 100 images
        :return: the Tensor if generated images is 'generated_images:0'
        """
        batch_z = tf.random_normal([100, self.noise_dim], name='noise_for_generation')
        self.__call__(batch_z, training=False, name='generated_images')

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')


class DiscriminatorDcgan:
    def __init__(self, first_layer_filters, conv_layers):
        """
        create the discriminator with the input conv2d layer numbers and first layer filter numbers.
        each layer has first_layer_filters*(2**(layer_index-1)) filters and no pooling are used.
        every conv2d layers with kernel_size=(5, 5), strides=[2, 2] and the activation is swish
        batch normalization is applied to every layer except the input layer
        the last layer is a fully connected layer with linear activation.
        all the kernel_initializer is tf.truncated_normal_initializer(stddev=0.02)
        :param first_layer_filters:  the numbers of filters in the first conv2d layer of the discriminator
        :param conv_layers: the numbers of conv2d layers of the discriminator
        """
        assert conv_layers >= 2, 'there must be more than 1 conv2d layers of the discriminator'
        # input layer, conv2d with no batch normalization
        self.input_layer = tf.layers.Conv2D(filters=first_layer_filters, kernel_size=(5, 5), strides=[2, 2],
                                            padding='same', data_format="channels_first", activation=tf.nn.swish,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='discriminator/conv_1/conv2d')
        self.conv_batch_norm_layers = []  # conv2d and batch normalization from the second layer
        for layer in range(2, conv_layers + 1):
            conv = tf.layers.Conv2D(filters=(2 ** (layer - 1)) * first_layer_filters, kernel_size=(5, 5), strides=[2, 2],
                                    padding='same', data_format="channels_first", use_bias=False,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='discriminator/conv_%d/conv2d' % layer)
            batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, name='discriminator/conv_%d/batch_normalization' % layer)
            name = 'discriminator/conv_%d/swish' % layer
            self.conv_batch_norm_layers.append((conv, batch_norm, name))
        self.output_layer_flatten = tf.layers.Flatten(name='discriminator/output/flatten')
        self.output_layer_fc = tf.layers.Dense(units=1, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name="discriminator/output/fc")
        self.output_layer_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, name='discriminator/output/batch_normalization')

    def __call__(self, batch, training):
        """
        discriminate the input images, output the batch of non-normalized probabilities
        """
        batch = self.input_layer(batch)
        for (conv, batch_norm, name) in self.conv_batch_norm_layers:
            batch = conv(batch)
            batch = batch_norm(batch, training=training)
            batch = tf.nn.swish(batch, name=name)
        batch = self.output_layer_flatten(batch)
        batch = self.output_layer_fc(batch)
        batch = self.output_layer_batch_norm(batch, training=training)
        return batch

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
