# -*- encoding: utf-8 -*-
# @Software: PyCharm
# @File    : celeba_main.py
# @Time    : 2019/1/23 11:41
# @Author  : LU Tianle

"""
"""
import os

import numpy as np
import tensorflow as tf
from PIL import Image

import components
import dcgan_nets
import inception_trans_nets
from gan import Gan

# ==============================================================================
DATASET = 'MNIST'  # 'MNIST' or 'Fashion MNIST'
# networks hyper parameters: details in dcgan_nets.py

# DCGAN Generator parameters:
GEN_CONV_FIRST_LAYER_FILTERS = 128
GEN_CONV_LAYERS = 3

# DCGAN Discriminator parameters:
DISC_FIRST_LAYER_FILTERS = 64
DISC_CONV_LAYERS = 3

GENERATOR_TYPE = 'DCGAN'  # 'DCGAN' or 'Inception-trans Nets'

# hyper-parameters:
BATCH_SIZE = 50
EPOCHS = 300
NOISE_DIM = 128

# vanilla gan training hyper-parameters
# DISCRIMINATOR_TRAINING_LOOP = 1
# GENERATOR_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, name='generator_optimizer_adam')
# DISCRIMINATOR_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, name='discriminator_optimizer_adam')
# TRAINING_ALGORITHM = "vanilla"

# wgan and sn-wgan training hyper-parameters
DISCRIMINATOR_TRAINING_LOOP = 5
GENERATOR_OPTIMIZER = tf.train.RMSPropOptimizer(learning_rate=5e-5, name='generator_optimizer_RMSProp')
DISCRIMINATOR_OPTIMIZER = tf.train.RMSPropOptimizer(learning_rate=5e-5, name='discriminator_optimizer_RMSProp')
# TRAINING_ALGORITHM = "wgan"
TRAINING_ALGORITHM = "sn-wgan"

# other parameters: details in gan.py
SAVE_PATH = os.getcwd() + os.path.sep + 'wgan'
CONTINUE_TRAINING = False
IMAGES_PER_ROW = 10

# generate images by the saved check points:
OUTPUT_IMAGE_PATH = os.getcwd() + os.path.sep + 'generated_images'
IMAGE_PAGES = 5
IMAGES_PER_ROW_FOR_GENERATING = 10

# training or inference
TRAINING_OR_INFERENCE = 'training'  # 'training' or 'inference'


# ==============================================================================


def celeba2tfrecord(img_dir, save_dir, size):
    assert not os.path.exists(os.path.join(save_dir, 'celeba_%d.tfrecord' % size)), 'tfrecord file already exist'
    components.images2tfrecord(img_dir, save_dir, name='celeba', size=size)


def get_celeba_dataset(tfrecord_dir):
    return tf.data.TFRecordDataset(tfrecord_dir, compression_type="ZLIB") \
        .map(components.parse_example) \
        .map(lambda img: tf.reshape(img, [3, 64, 64])) \
        .shuffle(30000)


def show_celeba_from_tfrecord(images_per_row):
    dataset_path = os.path.join('./data', 'celeba_64.tfrecord')
    assert os.path.exists(dataset_path), 'tfrecord file is not exist'
    dataset = get_celeba_dataset(dataset_path).batch(images_per_row ** 2)
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    iterator_initialize = iterator.make_initializer(dataset)
    [_, _, height, width] = dataset.output_shapes
    fig = np.zeros([height * images_per_row, width * images_per_row, 3]).astype('uint8')
    with tf.Session() as sess:
        sess.run(iterator_initialize)
        images = sess.run(iterator.get_next())
        images = np.around(np.transpose(images, [0, 2, 3, 1]) * 127.5 + 127.5).astype('uint8')
    for i in range(images_per_row):
        for j in range(images_per_row):
            fig[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = images[i * images_per_row + j]
    Image.fromarray(fig).show()


if __name__ == '__main__':
    # celeba2tfrecord(r'E:\tmp\img_align_celeba', save_dir=os.path.join('./', 'data'), size=64)  # convert images to a tfrecord file
    # show_celeba_from_tfrecord(10)

    # construct the networks and training algorithm
    celeba_dataset = get_celeba_dataset(os.path.join('./data', 'celeba_64.tfrecord'))
    image_shape = celeba_dataset.output_shapes.as_list()
    if GENERATOR_TYPE == 'DCGAN':
        generator = dcgan_nets.Generator(image_shape=image_shape, noise_dim=NOISE_DIM, first_conv_trans_layer_filters=GEN_CONV_FIRST_LAYER_FILTERS,
                                         conv_trans_layers=GEN_CONV_LAYERS)  # DCGAN Generator
    elif GENERATOR_TYPE == 'Inception-trans Nets':
        generator = inception_trans_nets.Generator(image_shape=image_shape, noise_dim=NOISE_DIM)  # Inception-trans Generator
    else:
        raise ValueError("Unknown Generator type")
    spectral_norm = True if TRAINING_ALGORITHM == 'sn-wgan' else False
    discriminator = dcgan_nets.Discriminator(first_layer_filters=DISC_FIRST_LAYER_FILTERS, conv_layers=DISC_CONV_LAYERS, spectral_norm=spectral_norm)
    gan = Gan(generator=generator, discriminator=discriminator, save_path=SAVE_PATH, noise_dim=NOISE_DIM)

    # training or inference
    if TRAINING_OR_INFERENCE == 'training':
        gan.train(dataset=celeba_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, discriminator_training_loop=DISCRIMINATOR_TRAINING_LOOP,
                  discriminator_optimizer=DISCRIMINATOR_OPTIMIZER, generator_optimizer=GENERATOR_OPTIMIZER, algorithm=TRAINING_ALGORITHM,
                  images_per_row=IMAGES_PER_ROW, continue_training=CONTINUE_TRAINING)
    elif TRAINING_OR_INFERENCE == 'inference':
        Gan.generate_image(save_path=SAVE_PATH, image_pages=IMAGE_PAGES, images_per_row=IMAGES_PER_ROW_FOR_GENERATING)
    else:
        raise ValueError("training or inference?")
