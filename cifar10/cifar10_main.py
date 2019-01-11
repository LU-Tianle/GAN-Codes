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
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import dcgan_nets
import components
from gan import Gan
import cv2

# ==============================================================================
DATASET = 'CIFAR-10'
# networks hyper parameters: details in dcgan_nets.py
GEN_CONV_FIRST_LAYER_FILTERS = 512
GEN_CONV_LAYERS = 3
DISC_FIRST_LAYER_FILTERS = 256
DISC_CONV_LAYERS = 3

# training hyper parameters:
BATCH_SIZE = 128
EPOCHS = 150
NOISE_DIM = 100

# vanilla gan training hyper-parameters
# DISCRIMINATOR_TRAINING_LOOP = 1
# GENERATOR_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2 * 1e-4, beta1=0.5, name='GENERATOR_OPTIMIZER_ADAM')
# DISCRIMINATOR_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2 * 1e-4, beta1=0.5, name='DISCRIMINATOR_OPTIMIZER_ADAM')
# TRAINING_ALGORITHM = "vanilla"

# wgan training hyper-parameters
DISCRIMINATOR_TRAINING_LOOP = 5
GENERATOR_OPTIMIZER = tf.train.RMSPropOptimizer(learning_rate=5e-5, name='generator_optimizer_RMSProp')
DISCRIMINATOR_OPTIMIZER = tf.train.RMSPropOptimizer(learning_rate=5e-5, name='discriminator_optimizer_RMSProp')
TRAINING_ALGORITHM = "wgan"

# other parameters: details in gan.py
SAVE_PATH = os.getcwd() + os.path.sep + 'wgan'
CONTINUE_TRAINING = False
INTERVAL_EPOCHS = 5
IMAGES_PER_ROW = 6

# generate images by saved check points:
OUTPUT_IMAGE_PATH = os.getcwd() + os.path.sep + 'generated_images'
IMAGE_PAGES = 5
IMAGES_PER_ROW_FOR_GENERATING = 6


# ==============================================================================


def get_cifar10_dataset(use_testset=False):
    """
    get mnist data and set the download path, data_format="channels_first"
    :return: (train_images, train_labels), (test_images, test_labels)
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    if use_testset:
        train_images = np.vstack((train_images, test_images))
        # normalize the images to the range of [-1, 1], the original range is {0, 1, ... , 255}
    train_images = ((train_images - 127.5) / 127.5).astype('float32')
    train_images = np.transpose(train_images, [0, 3, 2, 1])  # convert images to channel first
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size=60000)
    return train_dataset


def show_cifar10_pictures(images_per_row):
    """
    randomly show 6 and 3 pictures with their labels in the training set and test set respectively.
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = np.vstack((train_images, test_images))
    print(train_images.shape)
    train_images = cv2.cvtColor(train_images, cv2.COLOR_BGR2GRAY)
    fig = plt.figure(figsize=(images_per_row, images_per_row))
    fig.suptitle('images in CIFAR-10 dataset')
    for i in range(images_per_row ** 2):
        plt.subplot(images_per_row, images_per_row, i + 1)
        picture = random.randint(0, train_images.shape[0])
        plt.imshow(train_images[picture], cmap='gray')
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # show_cifar10_pictures(5)
    cifar10_dataset = get_cifar10_dataset(use_testset=True)
    # # construct the networks and training algorithm
    image_shape = cifar10_dataset.output_shapes.as_list()
    generator = dcgan_nets.Generator(image_shape=image_shape, first_conv_trans_layer_filters=GEN_CONV_FIRST_LAYER_FILTERS,
                                     conv_trans_layers=GEN_CONV_LAYERS, noise_dim=NOISE_DIM)
    discriminator = dcgan_nets.Discriminator(first_layer_filters=DISC_FIRST_LAYER_FILTERS, conv_layers=DISC_CONV_LAYERS)
    gan = Gan(generator=generator, discriminator=discriminator, save_path=SAVE_PATH)
    # training
    components.create_folder(SAVE_PATH, CONTINUE_TRAINING)
    components.save_parameters(first_conv_trans_layer_filters=GEN_CONV_FIRST_LAYER_FILTERS, conv_trans_layers=GEN_CONV_LAYERS,
                               first_layer_filters=DISC_FIRST_LAYER_FILTERS, conv_layers=DISC_CONV_LAYERS,
                               discriminator_training_loop=DISCRIMINATOR_TRAINING_LOOP,
                               dataset=DATASET, batch_size=BATCH_SIZE, noise_dim=NOISE_DIM, training_algorithm=TRAINING_ALGORITHM, save_path=SAVE_PATH)
    gan.train(dataset=cifar10_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, noise_dim=NOISE_DIM, discriminator_training_loop=DISCRIMINATOR_TRAINING_LOOP,
              discriminator_optimizer=DISCRIMINATOR_OPTIMIZER, generator_optimizer=GENERATOR_OPTIMIZER, algorithm=TRAINING_ALGORITHM,
              save_intervals=INTERVAL_EPOCHS, images_per_row=IMAGES_PER_ROW, continue_training=CONTINUE_TRAINING)
    # save parameters in save_path/parameter.txt"
    # # tensorboard --logdir=E:\workspace\GAN\mnist\saved_data_1
    # # localhost: 6006
    # # generate images using the latest saved check points and the images will be saved in 'save_path/images/'
    # # Gan.generate_image(save_path=SAVE_PATH, noise_dim=NOISE_DIM, image_pages=IMAGE_PAGES, images_per_row=IMAGES_PER_ROW_FOR_GENERATING)
