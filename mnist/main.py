# -*- encoding: utf-8 -*-
# @Software: PyCharm
# @File    : main.py 
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
import gan
import utils

# ==============================================================================
# networks hyper parameters: details in networks.py
GEN_CONV_TRANS_FIRST_LAYER_FILTERS = 64
GEN_CONV_TRANS_LAYERS = 2
DISC_FIRST_LAYER_FILTERS = 64
DISC_CONV_LAYERS = 2

# training hyper parameters:
BATCH_SIZE = 256
EPOCHS = 150
NOISE_DIM = 100
GENERATOR_OPTIMIZER = tf.train.AdamOptimizer(1e-4)
DISCRIMINATOR_OPTIMIZER = tf.train.AdamOptimizer(1e-4)
TRAINING_ALGORITHM = "vanilla"

# other parameters:
CHECK_POINTS_PATH = os.getcwd() + os.path.sep + 'checkpoints'  # the absolute path to save check points, it'll be created if not existed
CONTINUE_TRAINING = False  # continue training using the latest check points in the path above, if False all the files will be deleted in the folder
OUTPUT_IMAGE_PATH = os.getcwd() + os.path.sep + 'output'  # the absolute path to save output images, it'll be created if not existed
INTERVAL_EPOCHS = 1  # save check points every interval epochs
IMAGES_PER_ROW = 6


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
    utils.create_folder(CHECK_POINTS_PATH, CONTINUE_TRAINING)
    utils.create_folder(OUTPUT_IMAGE_PATH, CONTINUE_TRAINING)
    mnist_dataset = get_mnist_dataset(use_testset=True)
    image_shape = mnist_dataset.output_shapes.as_list()
    # dataset.output_shapes is a object of tf.TensorShape,
    # .as_list() Returns a list of integers or None for each dimension.
    generator = networks.GeneratorDcgan(image_shape=image_shape, first_conv_trans_layer_filters=GEN_CONV_TRANS_FIRST_LAYER_FILTERS,
                                        conv_trans_layers=GEN_CONV_TRANS_LAYERS)
    discriminator = networks.DiscriminatorDcgan(first_layer_filters=DISC_FIRST_LAYER_FILTERS, conv_layers=DISC_CONV_LAYERS)
    gan.Gan(generator=generator, discriminator=discriminator, check_points_path=CHECK_POINTS_PATH, output_image_path=OUTPUT_IMAGE_PATH) \
        .train(dataset=mnist_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, noise_dim=NOISE_DIM,
               discriminator_optimizer=DISCRIMINATOR_OPTIMIZER, generator_optimizer=GENERATOR_OPTIMIZER, algorithm=TRAINING_ALGORITHM,
               save_intervals=INTERVAL_EPOCHS, images_per_row=IMAGES_PER_ROW, continue_training=CONTINUE_TRAINING)
    # gan.generate_image(output_image_path=OUTPUT_IMAGE_PATH,
    #                    image_pages=FINAL_IMAGE_PAGES,
    #                    generate_midterm_images=GENERATE_MIDTERM_IMAGES)
