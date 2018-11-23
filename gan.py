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
import shutil


class Gan:
    def __init__(self, generator, discriminator, save_path):
        """
        construct the gan nets
        :param generator: the object of generator networks
        :param discriminator: the networks of discriminator networks
        :param save_path: check points and output image will be saved in folders in this path
        """
        self.generator = generator
        self.discriminator = discriminator
        self.save_path = save_path
        self.check_points_path = save_path + os.path.sep + 'check_points'
        self.output_image_path = save_path + os.path.sep + 'images_during_training'
        self.generator.generate()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    def train(self, dataset, batch_size, epochs, noise_dim,
              algorithm, discriminator_optimizer, generator_optimizer,
              save_intervals, images_per_row, continue_training=False):
        """
        start training and save models
        :param dataset: training dataset
        :param batch_size: mini batch size
        :param epochs: total training epochs
        :param noise_dim: noise dim for generation
        :param algorithm: training algorithm: 'vanilla' or 'wgan_gp'
        :param discriminator_optimizer: tf.train.Optimizer
        :param generator_optimizer: tf.train.Optimizer
        :param save_intervals: save check points and generated images every interval epochs in the self.save_path
        :param images_per_row: the number of generated images per row/column in a figure every interval epochs
        :param continue_training: continue training using the latest check points in the self.save_path
        """
        # TODO needs to add another training algorithm
        assert algorithm == "vanilla" or algorithm == "wgan_gp", 'illegal training algorithm'
        Gan.__create_folder(self.save_path, continue_training)
        Gan.__create_folder(self.check_points_path, continue_training)
        Gan.__create_folder(self.output_image_path, continue_training)
        # images will be generated after save_intervals epochs using the same noise
        random_vector_for_generation = tf.constant(np.random.randn(images_per_row ** 2, noise_dim), dtype=tf.float32, name='random_vector_for_generation')
        dataset = dataset.batch(batch_size, drop_remainder=True)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes, shared_name='training dataset iterator')
        iterator_initialize = iterator.make_initializer(dataset, name='training_dataset_iterator_initialize')
        generated_images = self.generator(tf.random_normal([batch_size, noise_dim], name='random_noise_for_training'), training=True)
        # feed images from training set and generator to discriminator
        discriminator_input = tf.concat([iterator.get_next(), generated_images], axis=0, name='discriminator_input')
        discriminator_output = self.discriminator(discriminator_input, training=True)
        real_output = tf.slice(discriminator_output, begin=[0, 0], size=[batch_size, -1], name='real_output')
        generated_output = tf.slice(discriminator_output, begin=[batch_size, 0], size=[batch_size, -1], name='generated_output')
        discriminator_loss, generator_loss = Gan.__loss(real_output, generated_output, algorithm=algorithm)
        train_discriminator = discriminator_optimizer.minimize(discriminator_loss, var_list=self.discriminator.var_list)
        train_generator = generator_optimizer.minimize(generator_loss, var_list=self.generator.var_list)
        generate_images_each_epoch = self.generator(random_vector_for_generation, training=False)
        saver = tf.train.Saver(max_to_keep=3)
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(self.save_path, sess.graph)  # save tensorboard files
            if continue_training:  # continue training from the latest check point
                latest_check_point_path = tf.train.latest_checkpoint(self.check_points_path)
                assert latest_check_point_path is not None, "no check points found"
                latest_check_point = int(latest_check_point_path.split("-")[1])
                saver.restore(sess, latest_check_point_path)
            else:  # start a initial training
                latest_check_point = 0
                sess.run(tf.global_variables_initializer())
            training_start_time = time.time()
            for epoch in range(epochs):  # training loop
                epoch_start_time = time.time()
                sess.run(iterator_initialize)
                try:
                    while True:
                        sess.run([train_discriminator, train_generator])
                except tf.errors.OutOfRangeError:
                    pass
                print('Time taken for epoch {} is {} sec'.format(epoch + latest_check_point + 1, time.time() - epoch_start_time))
                if (epoch + latest_check_point + 1) % save_intervals == 0:  # saving (checkpoint) the model every interval epochs
                    predictions = sess.run(generate_images_each_epoch)
                    Gan.__generate_and_save_images(self.output_image_path, predictions, images_per_row, epoch + latest_check_point + 1)
                    saver.save(sess=sess, save_path=os.path.join(self.check_points_path, 'check_points'), global_step=epoch + latest_check_point + 1)
            print('Time taken for training is {} min'.format((time.time() - training_start_time) / 60))
            writer.close()

    @staticmethod
    def generate_image(save_path, noise_dim, image_pages, images_per_row):
        """
        generate images using the latest saved check points and the images will be saved in 'save_path/images/'
        :param save_path: check points that have been saved in 'save_path/check_points/'
        :param image_pages: generated figure pages
        :param images_per_row: the number of generated images per row/column in a figure
        """
        assert images_per_row <= 10, 'too much images in a figure'
        check_points_path = os.path.join(save_path, 'check_points')
        output_image_path = os.path.join(save_path, 'images')
        Gan.__create_folder(output_image_path, False)
        latest_check_point_path = tf.train.latest_checkpoint(check_points_path)
        assert latest_check_point_path is not None, "no check points found"
        saver = tf.train.import_meta_graph(latest_check_point_path + '.meta')
        latest_check_point = int(latest_check_point_path.split("-")[1])
        shutil.copy(save_path + os.path.sep + "parameter.txt", output_image_path + os.path.sep + "parameter.txt")  # copy parameter text
        with tf.Session() as sess:
            saver.restore(sess, latest_check_point_path)
            for i in range(image_pages):
                generated_images = sess.run('generated_images:0')  # only generate 100 images each call
                Gan.__generate_and_save_images(output_image_path, generated_images, images_per_row, latest_check_point, i)

    @staticmethod
    def __loss(real_output, generated_output, algorithm):
        if algorithm == 'vanilla':
            real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
            generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)
            discriminator_loss = tf.add(real_loss, generated_loss, name='discriminator_loss')
            generator_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(generated_output), logits=generated_output)
            return discriminator_loss, generator_loss
        else:  # TODO needs to be implemented
            return 0, 0

    @staticmethod
    def __generate_and_save_images(output_path, predictions, images_per_row, epoch, index=0):
        fig = plt.figure(figsize=(images_per_row, images_per_row))
        fig.suptitle('epoch_{:04d}.png'.format(epoch))
        for i in range(images_per_row ** 2):
            plt.subplot(images_per_row, images_per_row, i + 1)
            if predictions.shape[1] == 1:
                plt.imshow(predictions[i][0] * 127.5 + 127.5, cmap='gray')
            else:
                plt.imshow(predictions[i] * 127.5 + 127.5)
            plt.axis('off')
        plt.savefig(output_path + os.path.sep + 'epoch_{:04d}_{}.png'.format(epoch, index))
        plt.close(fig)

    @staticmethod
    def __create_folder(path, continue_training=False):
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
