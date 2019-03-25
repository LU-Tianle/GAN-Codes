# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : mnist_main.py
# @Time    : 2018/11/23 21:09
# @Author  : LU Tianle

"""
the main function of training and generating cifar-10 images by gan
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
There are 50000 training images and 10000 test images.
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import tensorflow as tf
from PIL import Image

import components
import dcgan_nets
import inception_trans_nets
from gan import Gan

# ==============================================================================
DATASET = 'Cifar-10'
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
EPOCHS = 850
NOISE_DIM = 128
TRAINING_ALGORITHM = "vanilla"  # 'vanilla' 'wgan', 'sn-wgan'

if TRAINING_ALGORITHM == 'vanilla':  # vanilla gan training hyper-parameters
    DISCRIMINATOR_TRAINING_LOOP = 1
    GENERATOR_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, name='generator_optimizer_adam')
    DISCRIMINATOR_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, name='discriminator_optimizer_adam')
elif TRAINING_ALGORITHM == 'wgan' or TRAINING_ALGORITHM == 'sn-wgan':  # wgan and sn-wgan training hyper-parameters
    DISCRIMINATOR_TRAINING_LOOP = 5
    GENERATOR_OPTIMIZER = tf.train.RMSPropOptimizer(learning_rate=5e-5, name='generator_optimizer_RMSProp')
    DISCRIMINATOR_OPTIMIZER = tf.train.RMSPropOptimizer(learning_rate=5e-5, name='discriminator_optimizer_RMSProp')
else:
    raise ValueError("Unknown training algorithm")

# other parameters: details in gan.py
SAVE_PATH = os.getcwd() + os.path.sep + GENERATOR_TYPE + '_' + TRAINING_ALGORITHM
CONTINUE_TRAINING = False
IMAGES_PER_ROW = 10

# generate images by the saved check points:
OUTPUT_IMAGE_PATH = os.path.join(os.getcwd(), SAVE_PATH, 'generated_images')
IMAGE_PAGES = 5
IMAGES_PER_ROW_FOR_GENERATING = 10

# training or inference
TRAINING_OR_INFERENCE = 'training'  # 'training' or 'inference'


# ==============================================================================


def get_cifar10_dataset(tfrecord_dir=os.path.join('./data', 'cifar10.tfrecord')):
    return tf.data.TFRecordDataset(tfrecord_dir, compression_type="ZLIB") \
        .map(components.parse_example) \
        .map(lambda img: tf.reshape(img, [3, 32, 32])) \
        .shuffle(60000)


def show_cifar10_pictures_from_tfrecord(images_per_row):
    dataset_path = os.path.join('./data', 'cifar10.tfrecord')
    assert os.path.exists(dataset_path), 'tfrecord file is not exist'
    dataset = get_cifar10_dataset(dataset_path).batch(images_per_row ** 2)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    [_, _, height, width] = dataset.output_shapes
    fig = np.zeros([height * images_per_row, width * images_per_row, 3]).astype('uint8')
    with tf.Session() as sess:
        images = sess.run(next_element)
        images = np.around(np.transpose(images, [0, 2, 3, 1]) * 127.5 + 127.5).astype('uint8')
    for i in range(images_per_row):
        for j in range(images_per_row):
            fig[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = images[i * images_per_row + j]
    Image.fromarray(fig).show()


def cifar102tfrecord(save_dir):
    """
    :param save_dir:
    """
    assert not os.path.exists(os.path.join(save_dir, 'cifar10.tfrecord')), 'tfrecord file already exist'
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = np.vstack((train_images, test_images)).astype('float32')  # use test set to training
    train_images = ((train_images - 127.5) / 127.5)  # normalize the images to the range of [-1, 1], the original range is {0, 1, ... , 255}
    train_images = np.transpose(train_images, [0, 3, 1, 2])  # channel last to channel first
    writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(os.path.join(save_dir, 'cifar10.tfrecord'), writer_options)
    start_time = time.time()
    for i in range(len(train_images)):
        feature = tf.train.Features(feature={'img_bytes': __bytes_feature(train_images[i].tobytes())})
        example = tf.train.Example(features=feature)
        writer.write(example.SerializeToString())
        if (i + 1) % 1000 == 0:
            print('Time taken for {} images is {} min'.format((i + 1), (time.time() - start_time) / 60))
    print('images2tfrecord complete!')
    writer.close()


def __bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    # cifar102tfrecord(save_dir=os.path.join('./', 'data'))  # convert images to a tfrecord file
    # show_cifar10_pictures_from_tfrecord(10)

    # training or inference
    if TRAINING_OR_INFERENCE == 'training':
        # construct the networks and training algorithm
        cifar10_dataset = get_cifar10_dataset()
        image_shape = cifar10_dataset.output_shapes.as_list()
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
        gan.train(dataset=cifar10_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, discriminator_training_loop=DISCRIMINATOR_TRAINING_LOOP,
                  discriminator_optimizer=DISCRIMINATOR_OPTIMIZER, generator_optimizer=GENERATOR_OPTIMIZER, algorithm=TRAINING_ALGORITHM,
                  images_per_row=IMAGES_PER_ROW, continue_training=CONTINUE_TRAINING)
    elif TRAINING_OR_INFERENCE == 'inference':
        # tensorboard --logdir=E:\workspace\GAN\cifar10\...., localhost:6006
        # generate images using the latest saved check points and the images will be saved in 'save_path/images/'
        noise_list = [np.random.randn(100, NOISE_DIM) for i in range(IMAGE_PAGES)]
        Gan.generate_image(noise_list=noise_list, save_path=SAVE_PATH)
    else:
        raise ValueError("training or inference?")
