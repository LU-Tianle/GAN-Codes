# -*- encoding: utf-8 -*-
# @Software: PyCharm
# @File    : mnist_main.py
# @Time    : 2018/11/5 14:56
# @Author  : LU Tianle

"""
the main function of training and generating mnist images by gan
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import networks
import utils
from gan import Gan

# ==============================================================================
DATASET = 'MNIST'  # 'MNIST' or 'Fashion MNIST'
# networks hyper parameters: details in networks.py
GEN_CONV_FIRST_LAYER_FILTERS = 128
GEN_CONV_LAYERS = 2
DISC_FIRST_LAYER_FILTERS = 64
DISC_CONV_LAYERS = 2

# training hyper parameters:
BATCH_SIZE = 256
EPOCHS = 150
NOISE_DIM = 100
GENERATOR_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2 * 1e-4, beta1=0.5, name='GENERATOR_OPTIMIZER_ADAM')
DISCRIMINATOR_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2 * 1e-4, beta1=0.5, name='DISCRIMINATOR_OPTIMIZER_ADAM')
TRAINING_ALGORITHM = "vanilla"

# other parameters: details in gan.py
SAVE_PATH = os.getcwd() + os.path.sep + 'saved_data_1'
CONTINUE_TRAINING = False
INTERVAL_EPOCHS = 5
IMAGES_PER_ROW = 6

# generate images by saved check points:
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


def show_mnist_pictures():
    """
    randomly show 6 and 3 pictures with their labels in the training set and test set respectively.
    """
    (train_images, train_labels), (test_images, test_labels) = __get_mnist_data()
    figure, axes = plt.subplots(3, 3)
    for i in range(2):
        for j in range(3):
            picture = random.randint(0, train_images.shape[0])
            axes[i][j].imshow(train_images[picture][0], cmap='gray')
            axes[i][j].set_title("training set, label: " + str(train_labels[picture]))
            axes[i][j].axis('off')
    for i in range(3):
        picture = random.randint(0, test_images.shape[0])
        axes[2][i].imshow(test_images[picture][0], cmap='gray')
        axes[2][i].set_title("test set, label: " + str(test_labels[picture]))
        axes[2][i].axis('off')
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
    if DATASET == 'MNIST':
        mnist_dataset = get_mnist_dataset(use_testset=True)
    else:
        mnist_dataset = get_mnist_dataset(use_testset=True)
    # construct the networks and training algorithm
    image_shape = mnist_dataset.output_shapes.as_list()
    generator = networks.GeneratorDcgan(image_shape=image_shape, first_conv_trans_layer_filters=GEN_CONV_FIRST_LAYER_FILTERS, conv_trans_layers=GEN_CONV_LAYERS,
                                        noise_dim=NOISE_DIM)
    discriminator = networks.DiscriminatorDcgan(first_layer_filters=DISC_FIRST_LAYER_FILTERS, conv_layers=DISC_CONV_LAYERS)
    # gan = Gan(generator=generator, discriminator=discriminator, save_path=SAVE_PATH)
    # save parameters in save_path/parameter.txt"
    utils.save_parameters(first_conv_trans_layer_filters=GEN_CONV_FIRST_LAYER_FILTERS, conv_trans_layers=GEN_CONV_LAYERS,
                          first_layer_filters=DISC_FIRST_LAYER_FILTERS, conv_layers=DISC_CONV_LAYERS,
                          dataset=DATASET, batch_size=BATCH_SIZE, noise_dim=NOISE_DIM, training_algorithm=TRAINING_ALGORITHM, save_path=SAVE_PATH)
    # training
    # gan.train(dataset=mnist_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, noise_dim=NOISE_DIM,
    #           discriminator_optimizer=DISCRIMINATOR_OPTIMIZER, generator_optimizer=GENERATOR_OPTIMIZER, algorithm=TRAINING_ALGORITHM,
    #           save_intervals=INTERVAL_EPOCHS, images_per_row=IMAGES_PER_ROW, continue_training=CONTINUE_TRAINING)
    # tensorboard --logdir=E:\workspace\GAN\mnist\saved_data_1
    # localhost:6006
    # generate images using the latest saved check points and the images will be saved in 'save_path/images/'
    Gan.generate_image(save_path=SAVE_PATH, noise_dim=NOISE_DIM, image_pages=IMAGE_PAGES, images_per_row=IMAGES_PER_ROW_FOR_GENERATING)
