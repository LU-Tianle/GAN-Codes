# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : inception_trans_nets.py
# @Time    : 2019/1/5 14:31
# @Author  : LU Tianle

"""
Inception-trans-Res modules and nets
"""
import functools

import tensorflow as tf


class Generator:
    def __init__(self, image_shape, noise_dim):
        [self.channel, self.height, self.width] = image_shape
        if self.height == 28 and self.width == 28:
            self.project_shape = [-1, 3, 3, 256]
        elif self.height == 32 and self.width == 32:
            self.project_shape = [-1, 4, 4, 256]
        elif self.height == 64 and self.width == 64:
            self.project_shape = [-1, 4, 4, 512]
        else:
            raise ValueError("image shape incompatible")
        self.noise_dim = noise_dim
        self.project = tf.layers.Dense(units=functools.reduce(lambda x, y: abs(x) * y, self.project_shape), use_bias=False,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="generator/project/project")
        self.project_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='generator/project/bn')
        print("Inception-trans Nets Generator: ")
        print('    project and reshape: ' + str(self.project_shape))
        if (self.height == 28 and self.width == 28) or (self.height == 32 and self.width == 32):
            self.inception1 = InceptionTransRes1(ni_filter=[64], conv_trans_3x3_filters=[128, 64])  # 256 -> 128
            self.inception2 = InceptionTransRes2(ni_filter=[16], conv_trans_3x3_filters=[64, 16], conv_trans_5x5_filters=[128, 128, 32])  # 128 -> 64
            self.inception3 = InceptionTransRes3(ni_filter=[8], conv_trans_3x3_filters=[32, 8],
                                                 conv_trans_5x5_filters=[64, 64, 8], conv_trans_7x7_filters=[64, 64, 8])  # 64 -> 32
            self.inception4 = None
        elif self.height == 64 and self.width == 64:
            self.inception1 = InceptionTransRes1(ni_filter=[128], conv_trans_3x3_filters=[256, 128])  # 512 -> 256
            self.inception2 = InceptionTransRes2(ni_filter=[32], conv_trans_3x3_filters=[128, 32], conv_trans_5x5_filters=[256, 256, 64])  # 256 -> 128
            self.inception3 = InceptionTransRes3(ni_filter=[16], conv_trans_3x3_filters=[64, 16],
                                                 conv_trans_5x5_filters=[128, 128, 16], conv_trans_7x7_filters=[128, 128, 16])  # 128 -> 64
            self.inception4 = InceptionTrans4(ni_filter=[8], conv_trans_5x5_filters=[64, 64, 8], conv_trans_7x7_filters=[64, 64, 16])  # 64 -> 32
        else:
            raise ValueError("image shape incompatible")
        print('    conv_trans: filters=%d, No BN' % self.channel)
        self.conv_trans = tf.layers.Conv2DTranspose(filters=self.channel, kernel_size=(5, 5), padding='same',
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='generator/output_layer/conv')

    def __call__(self, batch_z, training, name):
        batch_z = self.project(batch_z)
        batch_z = self.project_bn(batch_z, training=training)
        batch_z = tf.nn.relu(batch_z, name=('generator/project/relu/' + name))
        batch_z = tf.reshape(batch_z, shape=self.project_shape, name=('generator/reshape/' + name))
        batch_z = self.inception1(batch_z, training=training, name=name)
        batch_z = self.inception2(batch_z, training=training, name=name)
        batch_z = self.inception3(batch_z, training=training, name=name)
        if self.inception4 is not None:
            batch_z = self.inception4(batch_z, training=training, name=name)
        batch_z = self.conv_trans(batch_z)
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


class InceptionTransRes1:
    def __init__(self, ni_filter, conv_trans_3x3_filters):
        """
        InceptionTransRes1
        :param ni_filters: 1-d vector: filters of the conv_1x1 before the nearest interpolation in the nearest interpolation branch
        :param conv_trans_3x3_filters: 2-d vector: filters of conv_tans_3x3 and the conv_1x1 in the conv_trans_3x3 branch
        """
        # branch1, nearest interpolation: conv1x1-bn-relu + NI
        print('    Inception-trans1: ')
        name = 'generator/inception_trans1/NI_branch/'
        print('        branch1: ')
        print('            conv1x1, filters=%d' % ni_filter[0])
        print('            NI, filters=%d' % ni_filter[0])
        self.branch1_conv1x1 = tf.layers.Conv2D(filters=ni_filter[0], kernel_size=[1, 1], padding='same', use_bias=False, name=name + 'conv1x1',
                                                data_format='channels_first', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.branch1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')
        # branch2, conv_trans_3x3 branch: conv_trans_3x3-bn-relu + conv_1x1-bn-relu
        name = 'generator/inception_trans1/conv_tans_3x3_branch/'
        print('        branch2: ')
        print('            conv_trans, filters=%d, kernel_size=(3, 3), strides=(2, 2)' % conv_trans_3x3_filters[0])
        print('            conv1x1, filters=%d' % conv_trans_3x3_filters[1])
        self.branch2_conv_tans_3x3 = tf.layers.Conv2DTranspose(filters=conv_trans_3x3_filters[0], kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                               use_bias=False, name=name + 'conv_tans_3x3',
                                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch2_conv_tans_3x3_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_bn')
        self.branch2_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_3x3_filters[1], kernel_size=(1, 1), padding='same', use_bias=False, name=name + 'conv_1x1',
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch2_conv_1x1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')

    def __call__(self, batch, training, name):
        # branch 1
        batch1 = self.branch1_conv1x1(batch)
        batch1 = self.branch1_batch_norm(batch1)
        batch1 = tf.nn.relu(batch1, name='generator/inception_trans1/NI_branch/relu/' + name)
        batch1 = tf.concat([batch1, batch1, batch1, batch1], name='generator/inception_trans1/NI_branch/NI_0/' + name)
        batch1 = tf.depth_to_space(batch1, 2, data_format="NCHW", name='generator/inception_trans1/NI_branch/NI_1/' + name)
        # branch 2
        batch2 = self.branch2_conv_tans_3x3(batch)
        batch2 = self.branch2_conv_tans_3x3_bn(batch2)
        batch2 = tf.nn.relu(batch2, name='generator/inception_trans1/conv_tans_3x3_branch/conv_tans_3x3_relu/' + name)
        batch2 = self.branch2_conv_1x1(batch2)
        batch2 = self.branch2_conv_1x1_bn(batch2)
        batch2 = tf.nn.relu(batch2, name='generator/inception_trans1/conv_tans_3x3_branch/conv_1x1_relu/' + name)
        # concatenate 2 batches, output feature map
        output = tf.concat([batch1, batch2], name='generator/inception_trans1/concatenate/' + name)
        if output.shape.as_list()[1:3] == [6, 6]:  # Padding for MNIST
            output = tf.pad(output, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]], mode='REFLECT', name='generator/inception_trans1/padding/' + name)
        return output


class InceptionTransRes2:
    def __init__(self, ni_filter, conv_trans_3x3_filters, conv_trans_5x5_filters):
        """
        InceptionTransRes2
        :param ni_filters: 1-d vector: filters of the conv_1x1 before the nearest interpolation in the nearest interpolation branch
        :param conv_trans_3x3_filters: 2-d vector: filters of conv_tans_3x3 and the conv_1x1 in the conv_trans_3x3 branch
        :param conv_trans_5x5_filters: 3-d vector: filters of conv_tans_3x3_1, conv_tans_3x3_2 and the conv_1x1 in the conv_trans_5x5 branch
        """
        # branch1, nearest interpolation: conv1x1-bn-relu + NI
        print('    Inception-trans2: ')
        print('        branch1: ')
        print('            conv1x1, filters=%d' % ni_filter[0])
        print('            NI, filters=%d' % ni_filter[0])
        name = 'generator/inception_trans2/NI_branch/'
        self.branch1_conv1x1 = tf.layers.Conv2D(filters=ni_filter[0], kernel_size=[1, 1], padding='same', use_bias=False, name=name + 'conv1x1',
                                                data_format='channels_first', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.branch1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')
        # branch2, conv_trans_3x3 block: conv_trans_3x3-batch normalization-relu + conv_1x1-batch normalization-relu
        name = 'generator/inception_trans2/conv_tans_3x3_branch/'
        print('        branch2: ')
        print('            conv_trans, filters=%d, kernel_size=(3, 3), strides=(2, 2)' % conv_trans_3x3_filters[0])
        print('            conv1x1, filters=%d' % conv_trans_3x3_filters[1])
        self.branch2_conv_tans_3x3 = tf.layers.Conv2DTranspose(filters=conv_trans_3x3_filters[0], kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                               use_bias=False, name=name + 'conv_tans_3x3',
                                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch2_conv_tans_3x3_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_bn')
        self.branch2_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_3x3_filters[1], kernel_size=(1, 1), padding='same', use_bias=False, name=name + 'conv_1x1',
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch2_conv_1x1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')
        # branch3, conv_trans_5x5 block: (conv_trans_3x3-batch normalization-relu)*2 + conv_1x1-batch normalization-relu
        name = 'generator/inception_trans2/conv_tans_5x5_branch/'
        print('        branch3: ')
        print('            conv_trans1, filters=%d, kernel_size=(3, 3), strides=(2, 2)' % conv_trans_5x5_filters[0])
        print('            conv_trans2, filters=%d, kernel_size=(3, 3), strides=(1, 1)' % conv_trans_5x5_filters[1])
        print('            conv1x1, filters=%d' % conv_trans_5x5_filters[2])
        self.branch3_conv_tans1 = tf.layers.Conv2DTranspose(filters=conv_trans_5x5_filters[0], kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                            use_bias=False, name=name + 'conv_tans_3x3_1',
                                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch3_conv_tans1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_1_bn')
        self.branch3_conv_tans2 = tf.layers.Conv2DTranspose(filters=conv_trans_5x5_filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                            use_bias=False, name=name + 'conv_tans_3x3_2',
                                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch3_conv_tan2_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_2_bn')
        self.branch3_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_5x5_filters[2], kernel_size=(1, 1), padding='same', use_bias=False, name=name + 'conv_1x1',
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch3_conv_1x1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')

    def __call__(self, batch, training, name):
        # shortcut
        shortcut = tf.concat([batch, batch, batch, batch], name='generator/inception_trans2/shortcut/NI_0/' + name)
        shortcut = tf.depth_to_space(shortcut, 2, data_format="NCHW", name='generator/inception_trans2/shortcut/NI_1/' + name)
        # branch 1
        batch1 = self.branch1_conv1x1(batch)
        batch1 = self.branch1_batch_norm(batch1)
        batch1 = tf.nn.relu(batch1, name='generator/inception_trans2/NI_branch/relu/' + name)
        batch1 = tf.concat([batch1, batch1, batch1, batch1], name='generator/inception_trans2/NI_branch/NI_0/' + name)
        batch1 = tf.depth_to_space(batch1, 2, data_format="NCHW", name='generator/inception_trans2/NI_branch/NI_1/' + name)
        # branch 2
        batch2 = self.branch2_conv_tans_3x3(batch)
        batch2 = self.branch2_conv_tans_3x3_bn(batch2)
        batch2 = tf.nn.relu(batch2, name='generator/inception_trans2/conv_tans_3x3_branch/conv_tans_3x3_relu/' + name)
        batch2 = self.branch2_conv_1x1(batch2)
        batch2 = self.branch2_conv_1x1_bn(batch2)
        batch2 = tf.nn.relu(batch2, name='generator/inception_trans2/conv_tans_3x3_branch/conv_1x1_relu/' + name)
        # branch3
        batch3 = self.branch3_conv_tans1(batch)
        batch3 = self.branch3_conv_tans1_bn(batch3)
        batch3 = tf.nn.relu(batch3, name='generator/inception_trans2/conv_tans_5x5_branch/conv_tans_3x3_1_relu/' + name)
        batch3 = self.branch3_conv_tans2(batch3)
        batch3 = self.branch3_conv_tan2_bn(batch3)
        batch3 = tf.add(batch3, shortcut, name='generator/inception_trans2/conv_tans_5x5_branch/add' + name)
        batch3 = tf.nn.relu(batch3, name='generator/inception_trans2/conv_tans_5x5_branch/conv_tans_3x3_2_relu/' + name)
        batch3 = self.branch3_conv_1x1(batch3)
        batch3 = self.branch3_conv_1x1_bn(batch3)
        batch3 = tf.nn.relu(batch3, name='generator/inception_trans2/conv_tans_5x5_branch/conv_1x1_relu/' + name)
        # concatenate 3 batches, output feature map
        output = tf.concat([batch1, batch2, batch3], name='generator/inception_trans2/concatenate/' + name)
        return output


class InceptionTransRes3:
    def __init__(self, ni_filter, conv_trans_3x3_filters, conv_trans_5x5_filters, conv_trans_7x7_filters):
        """
        InceptionTransRes2
        :param ni_filter: 1-d vector: filters of the conv_1x1 before the nearest interpolation in the nearest interpolation branch
        :param conv_trans_3x3_filters: 2-d vector: filters of conv_tans_3x3 and the conv_1x1 in the conv_trans_3x3 branch
        :param conv_trans_5x5_filters: 3-d vector: filters of conv_tans_3x3 and the conv_1x1 in the conv_trans_5x5 branch
        :param conv_trans_7x7_filters: 3-d vector: filters of conv_tans_1x7, conv_tans_7x1 and the conv_1x1 in the conv_trans_7x7 branch
        """
        # branch1, nearest interpolation: conv1x1-bn-relu + NI
        print('    Inception-trans3: ')
        print('        branch1: ')
        print('            conv1x1, filters=%d' % ni_filter[0])
        print('            NI, filters=%d' % ni_filter[0])
        name = 'generator/inception_trans3/NI_branch/'
        self.branch1_conv1x1 = tf.layers.Conv2D(filters=ni_filter[0], kernel_size=[1, 1], padding='same', use_bias=False, name=name + 'conv1x1',
                                                data_format='channels_first', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.branch1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')
        # branch2, conv_trans_3x3 block: conv_trans_3x3-batch normalization-relu + conv_1x1-batch normalization-relu
        name = 'generator/inception_trans3/conv_tans_3x3_branch/'
        print('        branch2: ')
        print('            conv_trans, filters=%d, kernel_size=(3, 3), strides=(2, 2)' % conv_trans_3x3_filters[0])
        print('            conv1x1, filters=%d' % conv_trans_3x3_filters[1])
        self.branch2_conv_tans_3x3 = tf.layers.Conv2DTranspose(filters=conv_trans_3x3_filters[0], kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                               use_bias=False, name=name + 'conv_tans_3x3',
                                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch2_conv_tans_3x3_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_bn')
        self.branch2_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_3x3_filters[1], kernel_size=(1, 1), padding='same', use_bias=False, name=name + 'conv_1x1',
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch2_conv_1x1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')
        # branch3, conv_trans_5x5 block: (conv_trans_3x3-batch normalization-relu)*2 + conv_1x1-batch normalization-relu
        name = 'generator/inception_trans3/conv_tans_5x5_branch/'
        print('        branch3: ')
        print('            conv_trans1, filters=%d, kernel_size=(3, 3), strides=(2, 2)' % conv_trans_5x5_filters[0])
        print('            conv_trans2, filters=%d, kernel_size=(3, 3), strides=(1, 1)' % conv_trans_5x5_filters[1])
        print('            conv1x1, filters=%d' % conv_trans_5x5_filters[2])
        self.branch3_conv_tans1 = tf.layers.Conv2DTranspose(filters=conv_trans_5x5_filters[0], kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                            use_bias=False, name=name + 'conv_tans_3x3_1',
                                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch3_conv_tans1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_1_bn')
        self.branch3_conv_tans2 = tf.layers.Conv2DTranspose(filters=conv_trans_5x5_filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                            use_bias=False, name=name + 'conv_tans_3x3_2',
                                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch3_conv_tan2_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_2_bn')
        self.branch3_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_5x5_filters[2], kernel_size=(1, 1), padding='same', use_bias=False, name=name + 'conv_1x1',
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch3_conv_1x1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')
        # branch4, conv_trans_7x7 block: conv_trans_1x7-bn-relu + conv_trans_7x1-bn-relu + conv_1x1-bn-relu
        name = 'generator/inception_trans3/conv_tans_7x7_branch/'
        print('        branch4: ')
        print('            conv_trans1, filters=%d, kernel_size=(1, 7), strides=(1, 2)' % conv_trans_7x7_filters[0])
        print('            conv_trans2, filters=%d, kernel_size=(7, 1), strides=(2, 1)' % conv_trans_7x7_filters[1])
        print('            conv1x1, filters=%d' % conv_trans_5x5_filters[2])
        self.branch4_conv_tans_1x7 = tf.layers.Conv2DTranspose(filters=conv_trans_7x7_filters[0], kernel_size=(1, 7), strides=(1, 2), padding='same',
                                                               use_bias=False, name=name + 'conv_tans_1x7',
                                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch4_conv_tans_1x7_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_1x7_bn')
        self.branch4_conv_tans_7x1 = tf.layers.Conv2DTranspose(filters=conv_trans_7x7_filters[1], kernel_size=(7, 1), strides=(2, 1), padding='same',
                                                               use_bias=False, name=name + 'conv_tans_7x1',
                                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch4_conv_tans_7x1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_7x1_bn')
        self.branch4_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_7x7_filters[2], kernel_size=(1, 1), padding='same', use_bias=False, name=name + 'conv_1x1',
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first")
        self.branch4_conv_1x1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')

    def __call__(self, batch, training, name):
        # shortcut
        shortcut = tf.concat([batch, batch, batch, batch], name='generator/inception_trans3/shortcut/NI_0/' + name)
        shortcut = tf.depth_to_space(shortcut, 2, data_format="NCHW", name='generator/inception_trans3/shortcut/NI_1/' + name)
        # branch 1
        batch1 = self.branch1_conv1x1(batch)
        batch1 = self.branch1_batch_norm(batch1)
        batch1 = tf.nn.relu(batch1, name='generator/inception_trans3/NI_branch/relu/' + name)
        batch1 = tf.concat([batch1, batch1, batch1, batch1], name='generator/inception_trans3/NI_branch/NI_0/' + name)
        batch1 = tf.depth_to_space(batch1, 2, data_format="NCHW", name='generator/inception_trans3/NI_branch/NI_1/' + name)
        # branch 2
        batch2 = self.branch2_conv_tans_3x3(batch)
        batch2 = self.branch2_conv_tans_3x3_bn(batch2)
        batch2 = tf.nn.relu(batch2, name='generator/inception_trans3/conv_tans_3x3_branch/conv_tans_3x3_relu/' + name)
        batch2 = self.branch2_conv_1x1(batch2)
        batch2 = self.branch2_conv_1x1_bn(batch2)
        batch2 = tf.nn.relu(batch2, name='generator/inception_trans3/conv_tans_3x3_branch/conv_1x1_relu/' + name)
        # branch3
        batch3 = self.branch3_conv_tans1(batch)
        batch3 = self.branch3_conv_tans1_bn(batch3)
        batch3 = tf.nn.relu(batch3, name='generator/inception_trans3/conv_tans_5x5_branch/conv_tans_3x3_1_relu/' + name)
        batch3 = self.branch3_conv_tans2(batch3)
        batch3 = self.branch3_conv_tan2_bn(batch3)
        batch3 = tf.add(batch3, shortcut, name='generator/inception_trans3/conv_tans_5x5_branch/add' + name)
        batch3 = tf.nn.relu(batch3, name='generator/inception_trans3/conv_tans_5x5_branch/conv_tans_3x3_2_relu/' + name)
        batch3 = self.branch3_conv_1x1(batch3)
        batch3 = self.branch3_conv_1x1_bn(batch3)
        batch3 = tf.nn.relu(batch3, name='generator/inception_trans3/conv_tans_5x5_branch/conv_1x1_relu/' + name)
        # branch4
        batch4 = self.branch4_conv_tans_1x7(batch)
        batch4 = self.branch4_conv_tans_1x7_bn(batch4)
        batch4 = tf.nn.relu(batch4, name='generator/inception_trans3/conv_tans_7x7_branch/conv_tans_1x7_relu/' + name)
        batch4 = self.branch4_conv_tans_7x1(batch4)
        batch4 = self.branch4_conv_tans_7x1_bn(batch4)
        batch4 = tf.add(batch4, shortcut, name='generator/inception_trans3/conv_tans_7x7_branch/add' + name)
        batch4 = tf.nn.relu(batch4, name='generator/inception_trans3/conv_tans_7x7_branch/conv_tans_7x1_relu/' + name)
        batch4 = self.branch4_conv_1x1(batch4)
        batch4 = self.branch4_conv_1x1_bn(batch4)
        batch4 = tf.nn.relu(batch4, name='generator/inception_trans3/conv_tans_7x7_branch/conv_1x1_relu/' + name)
        # concatenate 3 batches, output feature map
        output = tf.concat([batch1, batch2, batch3, batch4], name='generator/inception_trans3/concatenate/' + name)
        return output


class InceptionTrans4:
    def __init__(self, ni_filter, conv_trans_5x5_filters, conv_trans_7x7_filters):
        """
        InceptionTrans4
        :param ni_filter: 1-d vector: filters of the conv_1x1 before the nearest interpolation in the nearest interpolation branch
        :param conv_trans_5x5_filters: 3-d vector: filters of conv_tans_3x3 and the conv_1x1 in the conv_trans_5x5 branch
        :param conv_trans_7x7_filters: 3-d vector: filters of conv_tans_1x7, conv_tans_7x1 and the conv_1x1 in the conv_trans_7x7 branch
        """
        # branch1, nearest interpolation: conv1x1-bn-relu + NI
        print('    Inception-trans4: ')
        print('        branch1: ')
        print('            conv1x1, filters=%d' % ni_filter[0])
        print('            NI, filters=%d' % ni_filter[0])
        name = 'generator/inception_trans4/NI_branch/'
        self.branch1_conv1x1 = tf.layers.Conv2D(filters=ni_filter[0], kernel_size=[1, 1], padding='same', use_bias=False, name=name + 'conv1x1',
                                                data_format='channels_first', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.branch1_batch_norm = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')
        # branch2, conv_trans_5x5 block: (conv_trans_3x3-batch normalization-relu)*2 + conv_1x1-batch normalization-relu
        name = 'generator/inception_trans4/conv_tans_5x5_branch/'
        print('        branch2: ')
        print('            conv_trans1, filters=%d, kernel_size=(3, 3), strides=(2, 2)' % conv_trans_5x5_filters[0])
        print('            conv_trans2, filters=%d, kernel_size=(3, 3), strides=(1, 1)' % conv_trans_5x5_filters[1])
        print('            conv1x1, filters=%d' % conv_trans_5x5_filters[2])
        self.branch2_conv_tans1 = tf.layers.Conv2DTranspose(filters=conv_trans_5x5_filters[0], kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                            use_bias=False, name=name + 'conv_tans_3x3_1',
                                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch2_conv_tans1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_1_bn')
        self.branch2_conv_tans2 = tf.layers.Conv2DTranspose(filters=conv_trans_5x5_filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                            use_bias=False, name=name + 'conv_tans_3x3_2',
                                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch2_conv_tan2_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_3x3_2_bn')
        self.branch2_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_5x5_filters[2], kernel_size=(1, 1), padding='same', use_bias=False, name=name + 'conv_1x1',
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch2_conv_1x1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')
        # branch3, conv_trans_7x7 block: conv_trans_1x7-bn-relu + conv_trans_7x1-bn-relu + conv_1x1-bn-relu
        name = 'generator/inception_trans4/conv_tans_7x7_branch/'
        print('        branch3: ')
        print('            conv_trans1, filters=%d, kernel_size=(1, 7), strides=(1, 2)' % conv_trans_7x7_filters[0])
        print('            conv_trans2, filters=%d, kernel_size=(7, 1), strides=(2, 1)' % conv_trans_7x7_filters[1])
        print('            conv1x1, filters=%d' % conv_trans_5x5_filters[2])
        self.branch3_conv_tans_1x7 = tf.layers.Conv2DTranspose(filters=conv_trans_7x7_filters[0], kernel_size=(1, 7), strides=(1, 2), padding='same',
                                                               use_bias=False, name=name + 'conv_tans_1x7',
                                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch3_conv_tans_1x7_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_1x7_bn')
        self.branch3_conv_tans_7x1 = tf.layers.Conv2DTranspose(filters=conv_trans_7x7_filters[1], kernel_size=(7, 1), strides=(2, 1), padding='same',
                                                               use_bias=False, name=name + 'conv_tans_7x1',
                                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.branch3_conv_tans_7x1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_tans_7x1_bn')
        self.branch3_conv_1x1 = tf.layers.Conv2D(filters=conv_trans_7x7_filters[2], kernel_size=(1, 1), padding='same', use_bias=False, name=name + 'conv_1x1',
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format="channels_first")
        self.branch3_conv_1x1_bn = tf.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name + 'conv_1x1_bn')

    def __call__(self, batch, training, name):
        # shortcut
        shortcut = tf.concat([batch, batch, batch, batch], name='generator/inception_trans4/shortcut/NI_0/' + name)
        shortcut = tf.depth_to_space(shortcut, 2, data_format="NCHW", name='generator/inception_trans4/shortcut/NI_1/' + name)
        # branch 1
        batch1 = self.branch1_conv1x1(batch)
        batch1 = self.branch1_batch_norm(batch1)
        batch1 = tf.nn.relu(batch1, name='generator/inception_trans4/NI_branch/relu/' + name)
        batch1 = tf.concat([batch1, batch1, batch1, batch1], name='generator/inception_trans1/NI_branch/NI_0/' + name)
        batch1 = tf.depth_to_space(batch1, 2, data_format="NCHW", name='generator/inception_trans1/NI_branch/NI_1/' + name)
        # branch2
        batch2 = self.branch2_conv_tans1(batch)
        batch2 = self.branch2_conv_tans1_bn(batch2)
        batch2 = tf.nn.relu(batch2, name='generator/inception_trans4/conv_tans_5x5_branch/conv_tans_3x3_1_relu/' + name)
        batch2 = self.branch2_conv_tans2(batch2)
        batch2 = self.branch2_conv_tan2_bn(batch2)
        batch2 = tf.add(batch2, shortcut, name='generator/inception_trans4/conv_tans_5x5_branch/add' + name)
        batch2 = tf.nn.relu(batch2, name='generator/inception_trans4/conv_tans_5x5_branch/conv_tans_3x3_2_relu/' + name)
        batch2 = self.branch2_conv_1x1(batch2)
        batch2 = self.branch2_conv_1x1_bn(batch2)
        batch2 = tf.nn.relu(batch2, name='generator/inception_trans4/conv_tans_5x5_branch/conv_1x1_relu/' + name)
        # branch3
        batch3 = self.branch3_conv_tans_1x7(batch)
        batch3 = self.branch3_conv_tans_1x7_bn(batch3)
        batch3 = tf.nn.relu(batch3, name='generator/inception_trans4/conv_tans_7x7_branch/conv_tans_1x7_relu/' + name)
        batch3 = self.branch3_conv_tans_7x1(batch3)
        batch3 = self.branch3_conv_tans_7x1_bn(batch3)
        batch3 = tf.add(batch3, shortcut, name='generator/inception_trans4/conv_tans_7x7_branch/add' + name)
        batch3 = tf.nn.relu(batch3, name='generator/inception_trans4/conv_tans_7x7_branch/conv_tans_7x1_relu/' + name)
        batch3 = self.branch3_conv_1x1(batch3)
        batch3 = self.branch3_conv_1x1_bn(batch3)
        batch3 = tf.nn.relu(batch3, name='generator/inception_trans4/conv_tans_7x7_branch/conv_1x1_relu/' + name)
        # concatenate 3 batches, output feature map
        output = tf.concat([batch1, batch2, batch3], name='generator/inception_trans4/concatenate/' + name)
        return output
