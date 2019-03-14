# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : components.py
# @Time    : 2018/11/15 10:07
# @Author  : LU Tianle

"""
"""
import os
import shutil
import time
from glob import glob

import numpy as np
import scipy.misc
import tensorflow as tf


def create_folder(path, continue_training=False):
    """
    create a folder if it's not exist
    :param path: folder path,
    :param continue_training: if False, all folders and files in the path will be deleted recursively
    """
    if os.path.exists(path):
        if not continue_training:
            shutil.rmtree(path)  # Recursively delete a directory tree
            os.makedirs(path)
    else:
        os.makedirs(path)


def images2tfrecord(img_dir, save_dir, name, size=64):
    """
    convert images to a tfrecord file,from PIL import Image
    the images in the tfrecord file are resized, normalized to [-1,1) and channel first
    :param img_dir: images dir
    :param save_dir: the tfrecord file save dir
    :param name: tfrecord file name
    :param size: resize original images to height and width are [size, size]
    :return:
    """
    img_path_list = glob(os.path.join(img_dir, '*.jpg'))
    writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(os.path.join(save_dir, '%s_%d.tfrecord' % (name, size)), writer_options)
    start_time = time.time()
    for i in range(len(img_path_list)):
        img = scipy.misc.imread(img_path_list[i], mode='RGB')
        img = scipy.misc.imresize(img, [size, size])
        img = (img.astype('float32') - 127.5) / 127.5
        img = np.transpose(img, [2, 0, 1])
        feature = tf.train.Features(feature={'img_bytes': __bytes_feature(img.tobytes())})
        example = tf.train.Example(features=feature)
        writer.write(example.SerializeToString())
        if (i + 1) % 1000 == 0:
            print('Time taken for {} images is {} min'.format((i + 1), (time.time() - start_time) / 60))
    print('images2tfrecord complete!')
    writer.close()


def parse_example(serial_example):
    feats = tf.parse_single_example(serial_example, features={'img_bytes': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(feats['img_bytes'], tf.float32)
    return image


def __bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
