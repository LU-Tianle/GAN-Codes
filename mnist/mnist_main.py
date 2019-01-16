# -*- encoding: utf-8 -*-
# @Software: PyCharm
# @File    : mnist_main.py
# @Time    : 2018/11/5 14:56
# @Author  : LU Tianle

"""
the main function of training and generating mnist images by gan
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import components
import dcgan_nets
from gan import Gan

# ==============================================================================
DATASET = 'MNIST'  # 'MNIST' or 'Fashion MNIST'
# networks hyper parameters: details in dcgan_nets.py
GEN_CONV_FIRST_LAYER_FILTERS = 256
GEN_CONV_LAYERS = 3
DISC_FIRST_LAYER_FILTERS = 64
DISC_CONV_LAYERS = 3

# hyper-parameters:
BATCH_SIZE = 64
EPOCHS = 150
NOISE_DIM = 100

# vanilla gan training hyper-parameters
# DISCRIMINATOR_TRAINING_LOOP = 1
# GENERATOR_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, name='generator_optimizer_adam')
# DISCRIMINATOR_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, name='discriminator_optimizer_adam')
# TRAINING_ALGORITHM = "vanilla"

# wgan training hyper-parameters
DISCRIMINATOR_TRAINING_LOOP = 5
GENERATOR_OPTIMIZER = tf.train.RMSPropOptimizer(learning_rate=5e-5, name='generator_optimizer_RMSProp')
DISCRIMINATOR_OPTIMIZER = tf.train.RMSPropOptimizer(learning_rate=5e-5, name='discriminator_optimizer_RMSProp')
TRAINING_ALGORITHM = "wgan"
# TRAINING_ALGORITHM = "sn-wgan"

# other parameters: details in gan.py
SAVE_PATH = os.getcwd() + os.path.sep + 'wgan'
CONTINUE_TRAINING = False
INTERVAL_EPOCHS = 5
IMAGES_PER_ROW = 6

# generate images by the saved check points:
OUTPUT_IMAGE_PATH = os.getcwd() + os.path.sep + 'generated_images'
IMAGE_PAGES = 5
IMAGES_PER_ROW_FOR_GENERATING = 6


# ==============================================================================

def get_mnist_dataset(use_testset=False):
    """
    there are 60000 and 10000 labeled samples in the training set and test set respectively
    data_format="channels_first"
    :param use_testset: use test set as part of training set to train gan
    :return: tf.data.Dataset
    """
    (train_images, train_labels), (test_images, test_labels) = __get_mnist_data()
    if use_testset:
        train_images = np.vstack((train_images, test_images))
    # normalize the images to the range of [-1, 1], the original range is {0, 1, ... , 255}
    train_images = ((train_images - 127.5) / 127.5).astype('float32')
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size=70000)
    return train_dataset


def show_mnist_pictures(images_per_row):
    """
    randomly show 6 and 3 pictures with their labels in the training set and test set respectively.
    """
    (train_images, train_labels), (test_images, test_labels) = __get_mnist_data()
    train_images = np.vstack((train_images, test_images))
    fig = plt.figure(figsize=(images_per_row, images_per_row))
    fig.suptitle('images in MNIST dataset')
    for i in range(images_per_row ** 2):
        plt.subplot(images_per_row, images_per_row, i + 1)
        picture = random.randint(0, train_images.shape[0])
        plt.imshow(train_images[picture][0], cmap='gray')
        plt.axis('off')
    plt.show()


def __get_mnist_data():
    """
    get mnist data and set the download path, data_format="channels_first"
    :return: (train_images, train_labels), (test_images, test_labels)
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(
        os.getcwd() + os.path.sep + "mnistdata" + os.path.sep + "mnist.npz")
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
    return (train_images, train_labels), (test_images, test_labels)


if __name__ == '__main__':
    mnist_dataset = get_mnist_dataset(use_testset=True)
    # show_mnist_pictures(6)
    # construct the networks and training algorithm
    image_shape = mnist_dataset.output_shapes.as_list()
    generator = dcgan_nets.Generator(image_shape=image_shape, noise_dim=NOISE_DIM,
                                     first_conv_trans_layer_filters=GEN_CONV_FIRST_LAYER_FILTERS, conv_trans_layers=GEN_CONV_LAYERS)
    discriminator = dcgan_nets.Discriminator(first_layer_filters=DISC_FIRST_LAYER_FILTERS, conv_layers=DISC_CONV_LAYERS)
    gan = Gan(generator=generator, discriminator=discriminator, save_path=SAVE_PATH)
    components.create_folder(SAVE_PATH, CONTINUE_TRAINING)
    # save parameters in save_path/parameter.txt"
    components.save_parameters(first_conv_trans_layer_filters=GEN_CONV_FIRST_LAYER_FILTERS, conv_trans_layers=GEN_CONV_LAYERS, conv_layers=DISC_CONV_LAYERS,
                               first_layer_filters=DISC_FIRST_LAYER_FILTERS, discriminator_training_loop=DISCRIMINATOR_TRAINING_LOOP,
                               dataset=DATASET, batch_size=BATCH_SIZE, noise_dim=NOISE_DIM, training_algorithm=TRAINING_ALGORITHM, save_path=SAVE_PATH)
    # training
    gan.train(dataset=mnist_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, noise_dim=NOISE_DIM, discriminator_training_loop=DISCRIMINATOR_TRAINING_LOOP,
              discriminator_optimizer=DISCRIMINATOR_OPTIMIZER, generator_optimizer=GENERATOR_OPTIMIZER, algorithm=TRAINING_ALGORITHM,
              save_intervals=INTERVAL_EPOCHS, images_per_row=IMAGES_PER_ROW, continue_training=CONTINUE_TRAINING)
    # tensorboard --logdir=E:\workspace\GAN\mnist\saved_data_1
    # localhost:6006
    # generate images using the latest saved check points and the images will be saved in 'save_path/images/'
    # Gan.generate_image(save_path=SAVE_PATH, image_pages=IMAGE_PAGES, images_per_row=IMAGES_PER_ROW_FOR_GENERATING)
