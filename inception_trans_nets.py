# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : inception_trans_nets.py
# @Time    : 2019/1/5 14:31
# @Author  : LU Tianle

"""
inception-trans modules and nets
"""
import functools

import tensorflow as tf


class Generator:
    def __init__(self, image_shape, noise_dim):
        [self.channel, self.height, self.width] = image_shape
        self.noise_dim = noise_dim
        self.project_shape = [-1, 256, 4, 4]
        self.project = tf.layers.Dense(units=functools.reduce(lambda x, y: abs(x) * y, self.project_shape), use_bias=False,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="generator/project/project")
        self.project_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='generator/project/batch_norm')
        self.inception1 = InceptionTrans1(unpooling_filters=[64], conv_trans_3x3_filters=[256, 64])  # 256 -> 128
        self.inception2 = InceptionTrans2(unpooling_filters=[16], conv_trans_3x3_filters=[128, 16], conv_trans_5x5_filters=[128, 128, 32])  # 128 -> 64
        self.inception3 = InceptionTrans3(unpooling_filters=[8], conv_trans_5x5_filters=[64, 64, 8], conv_trans_7x7_filters=[64, 64, 16])  # 64 -> 32
        self.conv_trans = tf.layers.Conv2DTranspose(filters=self.channel, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format="channels_first",
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='generator/merge_channels/conv_1x1')

    def __call__(self, batch_z, training, name):
        batch_z = self.project(batch_z)
        batch_z = self.project_batch_norm(batch_z, training=training)
        batch_z = tf.nn.swish(batch_z, name=('generator/project/swish/' + name))
        batch_z = tf.reshape(batch_z, shape=self.project_shape, name=('generator/reshape/' + name))
        batch_z = self.inception1(batch_z, training=training, name=name)
        batch_z = self.inception2(batch_z, training=training, name=name)
        batch_z = self.inception3(batch_z, training=training, name=name)
        batch_z = self.conv_trans(batch_z)
        batch_z = tf.nn.tanh(batch_z, name=('generator/output/during_' + name))
        return batch_z

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')


class InceptionTrans1:
    def __init__(self, unpooling_filters, conv_trans_3x3_filters):
        """
        InceptionTrans1
        :param unpooling_filters: 1-d vector: filters of the conv_1x1 after the unpooling in unpooling branch
        :param conv_trans_3x3_filters: 2-d vector: filters of conv_tans_3x3 and the conv_1x1 in the conv_trans_3x3 branch
        """
        name = 'generator/inception_trans1/'
        # branch1, unpooling branch: unpooling + conv1x1-batch_normalization-swish
        # and these are equivalent conv_trans_1x1(strides=(2, 2))-batch_normalization-swish
        self.branch1_unpooling = tf.layers.Conv2DTranspose(filters=unpooling_filters[0], kernel_size=[1, 1], strides=(2, 2), padding='same',
                                                           use_bias=False, data_format='channels_first', name=name + 'unpooling_block/unpooling_and_conv1x1',
                                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.branch1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='generator/inception_trans1/unpooling_block/batch_norm')
        # branch2, conv_trans_3x3 branch: conv_trans_3x3-batch normalization-swish + conv_1x1-batch normalization-swish
        self.branch2_conv_tans = tf.layers.Conv2DTranspose(filters=conv_trans_3x3_filters[0], kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                           use_bias=False, name=name + 'conv_tans_3x3_block/conv_tans_3x3',
                                                           kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first")
        self.branch2_conv_tans_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_block/conv_tans_3x3_batch_norm')
        self.branch2_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_3x3_filters[1], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first",
                                                 name=name + 'conv_tans_3x3_block/conv_1x1')
        self.branch2_conv_1x1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_block/conv_1x1_batch_norm')

    def __call__(self, batch, training, name):
        # branch 1
        batch1 = self.branch1_unpooling(batch)
        batch1 = self.branch1_batch_norm(batch1)
        batch1 = tf.nn.swish(batch1, name='generator/inception_trans1/unpooling_block/swish/' + name)
        # branch 2
        batch2 = self.branch2_conv_tans(batch)
        batch2 = self.branch2_conv_tans_batch_norm(batch2)
        batch2 = tf.nn.swish(batch2, name='generator/inception_trans1/conv_tans_3x3_block/conv_tans_3x3_swish/' + name)
        batch2 = self.branch2_conv_1x1(batch2)
        batch2 = self.branch2_conv_1x1_batch_norm(batch2)
        batch2 = tf.nn.swish(batch2, name='generator/inception_trans1/conv_tans_3x3_block/conv_1x1_swish/' + name)
        # concatenate 2 batches, output feature map
        output = tf.concat([batch1, batch2], axis=1, name='generator/inception_trans1/concatenate/' + name)
        return output


class InceptionTrans2:
    def __init__(self, unpooling_filters, conv_trans_3x3_filters, conv_trans_5x5_filters):
        """
        InceptionTrans2
        :param unpooling_filters: 1-d vector: filters of the conv_1x1 after the unpooling in unpooling branch
        :param conv_trans_3x3_filters: 2-d vector: filters of conv_tans_3x3 and the conv_1x1 in the conv_trans_3x3 branch
        :param conv_trans_5x5_filters: 3-d vector: filters of conv_tans_3x3_1, conv_tans_3x3_2 and the conv_1x1 in the conv_trans_3x3 branch
        """
        name = 'generator/inception_trans2/'
        # branch1, unpooling branch: unpooling + conv1x1-batch_normalization-swish
        # and these are equivalent conv_trans_1x1(strides=(2, 2))-batch_normalization-swish
        self.branch1_unpooling = tf.layers.Conv2DTranspose(filters=unpooling_filters[0], kernel_size=[1, 1], strides=(2, 2), padding='same',
                                                           use_bias=False, name=name + 'unpooling_block/unpooling_and_conv1x1',
                                                           data_format='channels_first', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.branch1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'unpooling_block/batch_norm')
        # branch2, conv_trans_3x3 block: conv_trans_3x3-batch normalization-swish + conv_1x1-batch normalization-swish
        self.branch2_conv_tans_3x3 = tf.layers.Conv2DTranspose(filters=conv_trans_3x3_filters[0], kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                               use_bias=False, name=name + 'conv_tans_3x3_block/conv_tans_3x3',
                                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first")
        self.branch2_conv_tans_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_block/conv_tans_3x3_batch_norm')
        self.branch2_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_3x3_filters[1], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first",
                                                 name=name + 'conv_tans_3x3_block/conv_1x1')
        self.branch2_conv_1x1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_block/conv_1x1_batch_norm')
        # branch3, conv_trans_5x5 block: (conv_trans_3x3-batch normalization-swish)*2 + conv_1x1-batch normalization-swish
        self.branch3_conv_tans1 = tf.layers.Conv2DTranspose(filters=conv_trans_5x5_filters[0], kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                            use_bias=False, name=name + 'conv_tans_5x5_block/conv_tans_3x3_1',
                                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first")
        self.branch3_conv_tans1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9,
                                                                          name=name + 'conv_tans_5x5_block/conv_tans_3x3_1_batch_norm')
        self.branch3_conv_tans2 = tf.layers.Conv2DTranspose(filters=conv_trans_5x5_filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                            use_bias=False, name=name + 'conv_tans_5x5_block/conv_tans_3x3_2',
                                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first")
        self.branch3_conv_tan2_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9,
                                                                         name=name + 'conv_tans_5x5_block/conv_tans_3x3_2_batch_norm')
        self.branch3_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_5x5_filters[2], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first",
                                                 name=name + 'conv_tans_5x5_block/conv_1x1')
        self.branch3_conv_1x1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_5x5_block/conv_1x1_batch_norm')

    def __call__(self, batch, training, name):
        # branch 1
        batch1 = self.branch1_unpooling(batch)
        batch1 = self.branch1_batch_norm(batch1)
        batch1 = tf.nn.swish(batch1, name='generator/inception_trans2/unpooling_block/swish/' + name)
        # branch 2
        batch2 = self.branch2_conv_tans_3x3(batch)
        batch2 = self.branch2_conv_tans_batch_norm(batch2)
        batch2 = tf.nn.swish(batch2, name='generator/inception_trans2/conv_tans_3x3_block/conv_tans_3x3_swish/' + name)
        batch2 = self.branch2_conv_1x1(batch2)
        batch2 = self.branch2_conv_1x1_batch_norm(batch2)
        batch2 = tf.nn.swish(batch2, name='generator/inception_trans2/conv_tans_3x3_block/conv_1x1_swish/' + name)
        # branch3
        batch3 = self.branch3_conv_tans1(batch)
        batch3 = self.branch3_conv_tans1_batch_norm(batch3)
        batch3 = tf.nn.swish(batch3, name='generator/inception_trans2/conv_tans_5x5_block/conv_tans_3x3_1_swish/' + name)
        batch3 = self.branch3_conv_tans2(batch3)
        batch3 = self.branch3_conv_tan2_batch_norm(batch3)
        batch3 = tf.nn.swish(batch3, name='generator/inception_trans2/conv_tans_5x5_block/conv_tans_3x3_2_swish/' + name)
        batch3 = self.branch3_conv_1x1(batch3)
        batch3 = self.branch3_conv_1x1_batch_norm(batch3)
        batch3 = tf.nn.swish(batch3, name='generator/inception_trans2/conv_tans_5x5_block/conv_1x1_swish/' + name)
        # concatenate 3 batches, output feature map
        output = tf.concat([batch1, batch2, batch3], axis=1, name='generator/inception_trans2/concatenate/' + name)
        return output


class InceptionTrans3:
    def __init__(self, unpooling_filters, conv_trans_5x5_filters, conv_trans_7x7_filters):
        """
        InceptionTrans2
        :param unpooling_filters: 1-d vector: filters of the conv_1x1 after the unpooling in unpooling branch
        :param conv_trans_5x5_filters: 3-d vector: filters of conv_tans_3x3 and the conv_1x1 in the conv_trans_3x3 branch
        :param conv_trans_7x7_filters: 3-d vector: filters of conv_tans_1x7, conv_tans_7x1 and the conv_1x1 in the conv_trans_3x3 branch
        """
        name = 'generator/inception_trans3/'
        # branch1, unpooling branch: unpooling + conv1x1-batch_normalization-swish
        # and these are equivalent conv_trans_1x1(strides=(2, 2))-batch_normalization-swish
        self.branch1_unpooling = tf.layers.Conv2DTranspose(filters=unpooling_filters[0], kernel_size=[1, 1], strides=(2, 2), padding='same',
                                                           use_bias=False, name=name + 'unpooling_block/unpooling_and_conv1x1',
                                                           data_format='channels_first', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.branch1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'unpooling_block/batch_norm')
        # branch2, conv_trans_5x5 block: (conv_trans_3x3-batch normalization-swish)*2 + conv_1x1-batch normalization-swish
        self.branch2_conv_tans1 = tf.layers.Conv2DTranspose(filters=conv_trans_5x5_filters[0], kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                            use_bias=False, name=name + 'conv_tans_5x5_block/conv_tans_3x3_1',
                                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first")
        self.branch2_conv_tans1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9,
                                                                          name=name + 'conv_tans_5x5_block/conv_tans_3x3_1_batch_norm')
        self.branch2_conv_tans2 = tf.layers.Conv2DTranspose(filters=conv_trans_5x5_filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                            use_bias=False, name=name + 'conv_tans_5x5_block/conv_tans_3x3_2',
                                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first")
        self.branch2_conv_tan2_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9,
                                                                         name=name + 'conv_tans_5x5_block/conv_tans_3x3_2_batch_norm')
        self.branch2_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_5x5_filters[2], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first",
                                                 name=name + 'conv_tans_5x5_block/conv_1x1')
        self.branch2_conv_1x1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_5x5_block/conv_1x1_batch_norm')
        # branch3, conv_trans_7x7 block:
        # conv_trans_1x7-batch normalization-swish + conv_trans_7x1-batch normalization-swish + conv_1x1-batch normalization-swish
        self.branch3_conv_tans_1x7 = tf.layers.Conv2DTranspose(filters=conv_trans_7x7_filters[0], kernel_size=(1, 7), strides=(1, 2), padding='same',
                                                               use_bias=False, name=name + 'conv_tans_7x7_block/conv_tans_1x7',
                                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first")
        self.branch3_conv_tans_1x7_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9,
                                                                             name=name + 'conv_tans_7x7_block/conv_tans_1x7_batch_norm')
        self.branch3_conv_tans_7x1 = tf.layers.Conv2DTranspose(filters=conv_trans_7x7_filters[1], kernel_size=(7, 1), strides=(2, 1), padding='same',
                                                               use_bias=False, name=name + 'conv_tans_7x7_block/conv_tans_7x1',
                                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first")
        self.branch3_conv_tans_7x1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9,
                                                                             name=name + 'conv_tans_7x7_block/conv_tans_7x1_batch_norm')
        self.branch3_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_7x7_filters[2], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first",
                                                 name=name + 'conv_tans_7x7_block/conv_1x1')
        self.branch3_conv_1x1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_7x7_block/conv_1x1_batch_norm')

    def __call__(self, batch, training, name):
        # branch 1
        batch1 = self.branch1_unpooling(batch)
        batch1 = self.branch1_batch_norm(batch1)
        batch1 = tf.nn.swish(batch1, name='generator/inception_trans3/unpooling_block/swish/' + name)
        # branch2
        batch2 = self.branch2_conv_tans1(batch)
        batch2 = self.branch2_conv_tans1_batch_norm(batch2)
        batch2 = tf.nn.swish(batch2, name='generator/inception_trans3/conv_tans_5x5_block/conv_tans_3x3_1_swish/' + name)
        batch2 = self.branch2_conv_tans2(batch2)
        batch2 = self.branch2_conv_tan2_batch_norm(batch2)
        batch2 = tf.nn.swish(batch2, name='generator/inception_trans3/conv_tans_5x5_block/conv_tans_3x3_2_swish/' + name)
        batch2 = self.branch2_conv_1x1(batch2)
        batch2 = self.branch2_conv_1x1_batch_norm(batch2)
        batch2 = tf.nn.swish(batch2, name='generator/inception_trans3/conv_tans_5x5_block/conv_1x1_swish/' + name)
        # branch3
        batch3 = self.branch3_conv_tans_1x7(batch)
        batch3 = self.branch3_conv_tans_1x7_batch_norm(batch3)
        batch3 = tf.nn.swish(batch3, name='generator/inception_trans3/conv_tans_7x7_block/conv_tans_1x7_swish/' + name)
        batch3 = self.branch3_conv_tans_7x1(batch3)
        batch3 = self.branch3_conv_tans_7x1_batch_norm(batch3)
        batch3 = tf.nn.swish(batch3, name='generator/inception_trans3/conv_tans_7x7_block/conv_tans_7x1_swish/' + name)
        batch3 = self.branch3_conv_1x1(batch3)
        batch3 = self.branch3_conv_1x1_batch_norm(batch3)
        batch3 = tf.nn.swish(batch3, name='generator/inception_trans3/conv_tans_7x7_block/conv_1x1_swish/' + name)
        # concatenate 3 batches, output feature map
        output = tf.concat([batch1, batch2, batch3], axis=1, name='generator/inception_trans3/concatenate/' + name)
        return output
