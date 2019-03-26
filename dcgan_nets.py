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
    def __init__(self, image_shape, first_conv_trans_layer_filters, conv_trans_layers, noise_dim):
        """
        create the generator with the input conv2d_transpose layer numbers and first layer filter numbers.
        the first layer is a fully connected layer with batch normalization
            that projects and reshapes the random noise followed by several conv2d_transpose layers
        every conv2d_transpose layers has kernel_size=(5, 5) strides=[2, 2] and filters = first_conv_trans_layer_filters // (2 ** (layer - 1))
        the conv_1x1 with batch_norm layer is optional and it has half of the filters of it's corresponding conv2d_transpose
        all the kernel_initializer is tf.truncated_normal_initializer(stddev=0.02)
        batch normalization and ReLU activation is applied to every layer except the output layer
        :param image_shape: output image shape [height, width, channel]
        :param first_conv_trans_layer_filters: the numbers of filters in the first conv2d_transpose layer of the generator
        :param conv_trans_layers: the numbers of conv2d_transpose layers of the generator
        """
        [self.height, self.width, self.channel] = image_shape
        assert conv_trans_layers >= 2, 'there must be more than 2 conv2d_transpose layers of the generator'
        # the first layer is project and reshape the random noise
        self.noise_dim = noise_dim
        self.project_shape = [-1, self.height // (2 ** conv_trans_layers), self.width // (2 ** conv_trans_layers), 2 * first_conv_trans_layer_filters, ]
        self.project = tf.layers.Dense(units=functools.reduce(lambda x, y: abs(x) * y, self.project_shape), use_bias=False,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="generator/project/project")
        self.project_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='generator/project/batch_norm')
        self.conv_trans_batch_norm_layers = []  # conv2d transpose layers with batch normalization
        print("DCGAN Generator: ")
        print('    project and reshape: ' + str(self.project_shape))
        for layer in range(1, conv_trans_layers):
            layer_name = 'generator/conv_trans_%d/' % layer
            filters = first_conv_trans_layer_filters // (2 ** (layer - 1))
            print('    conv_trans_%d: filters=%d' % (layer, filters))
            conv_trans = tf.layers.Conv2DTranspose(filters=filters, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False,
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name=layer_name + 'conv_trans')
            batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=layer_name + 'conv_trans_batch_norm')
            self.conv_trans_batch_norm_layers.append((conv_trans, batch_norm, layer_name))
        # output layer whose output shape is [batch_size, height, width, channel], no batch normalization
        print('    conv_trans_%d: filters=%d, No BN' % (conv_trans_layers, self.channel))
        self.output_layer = tf.layers.Conv2DTranspose(filters=self.channel, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='generator/output_layer/conv_trans')

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
            if batch_z.shape.as_list()[1:3] == [6, 6]:  # MNIST
                batch_z = tf.pad(batch_z, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]], mode='REFLECT', name='generator/padding')
        batch_z = self.output_layer(batch_z)
        batch_z = tf.nn.tanh(batch_z, name=('generator/output_layer/tanh/during_' + name))
        return batch_z

    def generate(self):
        """
        used for generating images, generate 100 images
        :return: the Tensor if generated images is 'generator/output_layer/tanh/during_inference:0'
        """
        batch_z = tf.placeholder(dtype=tf.float32, shape=[None, self.noise_dim], name='noise_for_inference')
        self.__call__(batch_z, training=False, name='inference')

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')


class Discriminator:
    def __init__(self, first_layer_filters, conv_layers, spectral_norm, batch_norm, labels=None):
        """
        create the discriminator with the input conv2d layer numbers and first layer filter numbers.
        each layer has first_layer_filters*(2**(layer_index-1)) filters and no pooling are used.
        every conv2d layers has a kernel_size=(5, 5), strides=[2, 2] and a leakeyReLU activation
        batch normalization is applied to every layer except the first conv and conv_1x1 layer (if this conv_1x1 exist)
        the last layer is a fully connected layer with linear activation and no batch normalization.
        all the kernel_initializer is tf.truncated_normal_initializer(stddev=0.02)
        :param first_layer_filters:  the numbers of filters in the first conv2d layer of the discriminator
        :param conv_layers: the numbers of conv2d layers of the discriminator
        :param spectral_norm: use Spectral Normalization
        :param batch_norm:  use batch_norm Normalization
        :param labels: classification probability for ACGAN
        """
        print(batch_norm)
        assert conv_layers >= 2, 'there must be more than 1 conv2d layers of the discriminator'
        self.conv_batch_norm_layers = []
        print("DCGAN Discriminator: ")
        for layer in range(1, conv_layers + 1):
            layer_name = 'discriminator/conv_%d/' % layer
            filters = int((2 ** (layer - 1))) * first_layer_filters
            if spectral_norm:
                conv = Conv2D(filters=filters, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                              use_bias=False, name=layer_name + 'conv_5x5', spectral_norm=spectral_norm, padding='same')
            else:
                conv = tf.layers.Conv2D(filters=filters, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                        use_bias=False, padding='same', name=layer_name + 'conv_5x5')
            if layer == 1 or batch_norm is False:  # conv_1 has no batch normalization
                bn = None
                print('    conv_%d: filters=%d, No BN' % (layer, filters))
            else:
                bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=layer_name + 'conv_batch_norm')
                print('    conv_%d: filters=%d' % (layer, filters))
            self.conv_batch_norm_layers.append((conv, bn, layer_name))
        self.output_layer_flatten = tf.layers.Flatten(name='discriminator/output_layer/flatten')
        if spectral_norm:
            self.output_layer_fc = Dense(units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=True,
                                         spectral_norm=spectral_norm, name="discriminator/output_layer/fc")
        else:
            self.output_layer_fc = tf.layers.Dense(units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=True,
                                                   name="discriminator/output_layer/fc")
        print('    flatten and fc')
        if labels is not None:  # classification for ACGAN
            print('    flatten and fc for ACGAN, labels=%d' % labels)
            if spectral_norm:
                self.output_layer_classification = Dense(units=labels, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=True,
                                                         name="discriminator/output_layer/classification/fc")
            else:
                self.output_layer_classification = tf.layers.Dense(units=labels, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=True,
                                                                   name="discriminator/output_layer/classification/fc")
        else:
            self.output_layer_classification = None

    def __call__(self, batch, training, name):
        """
         discriminate the input images, output the batch of non-normalized probabilities
        :param batch:
        :param training:
        :param name: training_generator or training_discriminator
        :return:
        """
        for (conv, bn, layer_name) in self.conv_batch_norm_layers:
            batch = conv(batch)
            if bn is not None:  # the first conv layer has no batch norm
                batch = bn(batch, training=training)
            batch = tf.nn.leaky_relu(batch, name='%sconv_5x5_activation/%s' % (layer_name, name))
        batch = self.output_layer_flatten(batch)
        output = self.output_layer_fc(batch)
        if self.output_layer_classification is not None:
            probabilities = self.output_layer_classification(batch)
            probabilities = tf.nn.sigmoid(probabilities, name='discriminator/output_layer/classification/sigmoid')
        else:
            probabilities = None
        return output, probabilities

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
