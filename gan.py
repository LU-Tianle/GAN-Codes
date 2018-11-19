# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : gan.py
# @Time    : 2018/11/12 20:49
# @Author  : LU Tianle

"""
training algorithms of gan
"""

import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Gan:
    def __init__(self, generator, discriminator, check_points_path, output_image_path):
        """
        construct the gan nets
        :param generator: the object of generator networks
        :param discriminator: the networks of discriminator networks
        :param check_points_path: check points path
        :param output_image_path: output image path
        """
        self.generator = generator
        self.discriminator = discriminator
        self.check_points_path = check_points_path
        self.output_image_path = output_image_path

    def train(self, dataset, batch_size, epochs, noise_dim,
              algorithm, discriminator_optimizer, generator_optimizer,
              save_intervals, continue_training=False):
        """
        start training and save models
        :param dataset: training dataset
        :param batch_size: mini batch size
        :param epochs: total training epochs
        :param noise_dim: noise dim
        :param algorithm: training algorithm: 'vanilla' or 'wgan_gp'
        :param discriminator_optimizer: tf.train.Optimizer
        :param generator_optimizer: tf.train.Optimizer
        :param save_intervals: save check points and generate images every interval epochs
        :param continue_training: continue training using the latest check points
        """
        # TODO needs to add another training algorithm
        assert algorithm == "vanilla" or algorithm == "wgan_gp", 'illegal training algorithm'
        dataset = dataset.batch(batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        iterator_initialize = iterator.make_initializer(dataset)
        random_vector_for_generation = tf.constant(np.random.randn(25, noise_dim))
        generated_images = self.generator(tf.random_normal([batch_size, noise_dim]), training=True)
        discriminator_input = tf.concat([iterator.get_next(), generated_images], axis=0)  # feed images from training set and generator to discriminator
        discriminator_output = self.discriminator(discriminator_input, training=True)
        real_output = discriminator_output[:batch_size]
        generated_output = discriminator_output[batch_size:]
        discriminator_loss, generator_loss = Gan.__loss(real_output, generated_output, algorithm=algorithm)
        train_discriminator = discriminator_optimizer.minimize(discriminator_loss, var_list=self.discriminator.var_list)
        train_generator = generator_optimizer.minimize(generator_loss, var_list=self.generator.var_list)
        # images will be generated after each epoch using the same noise
        # generate_images_each_epoch = self.generator(random_vector_for_generation, training=False)
        saver = tf.train.Saver(max_to_keep=0)
        with tf.Session() as sess:
            # if continue_training:  # continue training from the latest check point
            #     latest_check_point = self.__latest_check_point_epoch()
            #     self.saver.restore(sess, tf.train.latest_checkpoint(self.check_points_path))
            # else:  # start a initial training
            latest_check_point = 0
            sess.run(tf.global_variables_initializer())
            training_start_time = time.time()
            # random_vector_for_generation = tf.random_normal([16, noise_dim]).eval()
            for epoch in range(epochs):  # training loop
                epoch_start_time = time.time()
                sess.run(iterator_initialize)
                try:
                    while True:
                        sess.run([train_discriminator, train_generator])
                except tf.errors.OutOfRangeError:
                    pass
                print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - epoch_start_time))
                # if (epoch + latest_check_point + 1) % save_intervals == 0:  # saving (checkpoint) the model every interval epochs
                    # predictions = sess.run(generate_images_each_epoch)
                    # self.__generate_and_save_images(predictions, epoch + 1)
                    #     # every check points files will be saved in folder '.../checkpoints/epoch_num'
                    #     epoch_check_points_path = self.check_points_path.join(os.path.sep + "epoch_" + str(epoch + latest_check_point + 1))
                    #     utils.create_folder(path=epoch_check_points_path, clean_folder=True)
                    # saver.save(sess=sess, save_path=self.check_points_path, global_step=epoch + latest_check_point + 1)
            print('Time taken for training is {} min'.format((time.time() - training_start_time) / 60))

    # @staticmethod
    # def generate_image(output_image_path, image_pages, generate_midterm_images=True, use_same_noise=True):
    #     """
    #     generate images use saved check points
    #     :param output_image_path:
    #     :param image_pages:
    #     :param generate_midterm_images:
    #     :param use_same_noise:
    #     :return:
    #     """
    #     with tf.Session() as sess:
    #         saver.restore(sess, check_points_path)

    @staticmethod
    def __loss(real_output, generated_output, algorithm):
        if algorithm == 'vanilla':
            real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
            generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)
            discriminator_loss = real_loss + generated_loss
            generator_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(generated_output), logits=generated_output)
            return discriminator_loss, generator_loss
        else:  # TODO needs to be implemented
            return 0, 0

    def __generate_and_save_images(self, predictions, epoch):
        plt.figure(figsize=(5, 5))
        for i in range(predictions.shape[0]):
            plt.subplot(5, 5, i + 1)
            plt.imshow(predictions[i][0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.savefig(self.output_image_path + os.path.sep + 'image_epoch_{:04d}.png'.format(epoch))
