# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : gan.py
# @Time    : 2018/11/12 20:49
# @Author  : LU Tianle

"""
training algorithms of vanilla gan
"""
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import components


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
        self.check_points_path = os.path.join(save_path, 'check_points')
        self.output_image_path = os.path.join(save_path, 'images_during_training')
        self.generator.generate()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    def train(self, dataset, batch_size, epochs, noise_dim, algorithm,
              discriminator_training_loop, discriminator_optimizer, generator_optimizer,
              save_intervals, images_per_row, continue_training=False):
        """
        start training and save models
        :param dataset: training dataset
        :param batch_size: mini batch size
        :param epochs: total training epochs
        :param noise_dim: noise dim for generation
        :param algorithm: 'vanilla or wgan'
        :param discriminator_training_loop: update generator once while updating discriminator more times
        :param discriminator_optimizer: tf.train.Optimizer
        :param generator_optimizer: tf.train.Optimizer
        :param save_intervals: save generated images every interval epochs in the self.save_path while save check points every (2*interval epochs)
        :param images_per_row: the number of generated images per row/column in a figure every interval epochs
        :param continue_training: continue training using the latest check points in the self.save_path
        """
        components.create_folder(self.check_points_path, continue_training)
        components.create_folder(self.output_image_path, continue_training)
        # saved training iterations
        saved_iterations = tf.get_variable(name='saved_iterations', dtype=tf.int32, initializer=0, trainable=False)
        # create training dataset
        with tf.name_scope('dataset'):
            dataset = dataset.batch(batch_size, drop_remainder=True)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            iterator_initialize = iterator.make_initializer(dataset)
        noise_for_training = tf.random_normal([batch_size, noise_dim], name='noise_for_training')
        # discriminator loss
        generated_images = self.generator(noise_for_training, training=False, name='training_discriminator')
        discriminator_input = tf.concat([iterator.get_next(), generated_images], axis=0, name='discriminator_input')
        discriminator_output = self.discriminator(discriminator_input, training=True, name='training_discriminator')
        discriminator_loss = Gan.__discriminator_loss(discriminator_output, batch_size, algorithm)
        # generator loss
        generated_images = self.generator(noise_for_training, training=True, name='training_generator')
        discriminator_output = self.discriminator(generated_images, training=False, name='training_generator')
        generator_loss = Gan.__generator_loss(discriminator_output, algorithm)
        # save tensorboard files
        with tf.name_scope('tensorboard_summary'):
            if algorithm == 'vanilla':
                tf.summary.scalar(name='JS_divergence', tensor=-0.5 * discriminator_loss + 1)
            elif algorithm == 'wgan' or algorithm == 'sn-wgan':
                tf.summary.scalar(name='Wasserstein_distance', tensor=-discriminator_loss)
            else:
                raise ValueError('unknown input training algorithm')
            summaries = tf.summary.merge_all()
        # optimization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):  # update the population statistics of batch normalization before training
            train_discriminator = discriminator_optimizer.minimize(discriminator_loss, var_list=self.discriminator.var_list)
            train_generator = generator_optimizer.minimize(generator_loss, var_list=self.generator.var_list)
        # weight clipping in wgan
        if algorithm == 'wgan':
            with tf.name_scope('discriminator/clipping'):
                clipping_vars = []
                for var in self.discriminator.var_list:
                    if 'batch_normalization' not in var.name:
                        clipping_vars.append(var)
                clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01, name='clip/' + var.op.name), name='update/' + var.op.name) for var in clipping_vars]
        else:
            clip = tf.no_op(name='no_op')
        # images will be generated after save_intervals epochs using the same noise
        noise_for_generation = tf.get_variable(name='noise_for_generation', shape=[images_per_row ** 2, noise_dim],
                                               initializer=tf.random_normal_initializer, trainable=False)
        generate_images_each_epoch = self.generator(noise_for_generation, training=False, name='inference_epoch')
        saver = tf.train.Saver(max_to_keep=3)
        with tf.Session() as sess:
            if continue_training:  # continue training from the latest check point
                latest_training_epochs_path = tf.train.latest_checkpoint(self.check_points_path)
                assert latest_training_epochs_path is not None, "no check points found"
                latest_training_epochs = int(latest_training_epochs_path.split("-")[1])  # load training epochs
                saver.restore(sess, latest_training_epochs_path)
                iterations = sess.run('saved_iterations:0')  # load training iterations
            else:  # start a initial training
                latest_training_epochs = 0
                iterations = 0  # training mini-batch iterations
                sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(self.save_path, sess.graph)
            training_start_time = time.time()
            for epoch in range(epochs):  # training loop
                epoch_start_time = time.time()
                sess.run(iterator_initialize)
                save_summary = 0
                try:
                    while True:
                        if algorithm == 'wgan' or algorithm == 'sn-wgan':  # training original wgan
                            loop = 5 * discriminator_training_loop if iterations < 25 or (iterations + 1) % 500 == 0 else discriminator_training_loop
                            for i in range(loop):  # wgan training trick: do more discriminator update occasionally or at the beginning
                                if i == loop - 1:
                                    _, save_summary = sess.run([train_discriminator, summaries])
                                else:
                                    sess.run([train_discriminator])
                                if algorithm == 'wgan':  # wgan clipping
                                    sess.run([clip])
                        else:
                            _, save_summary = sess.run([train_discriminator, summaries])
                        writer.add_summary(save_summary, iterations)
                        iterations += 1
                        sess.run([train_generator])
                except tf.errors.OutOfRangeError:
                    pass
                print('Time taken for epoch {} is {} sec'.format(epoch + latest_training_epochs + 1, time.time() - epoch_start_time))
                if (epoch + latest_training_epochs + 1) % save_intervals == 0:
                    predictions = sess.run(generate_images_each_epoch)
                    # saving the model(checkpoint) and generated images
                    Gan.__save_images(self.output_image_path, predictions, images_per_row, epoch + latest_training_epochs + 1)
                    if (epoch + latest_training_epochs + 1) % (2 * save_intervals) == 0:
                        sess.run(tf.assign(saved_iterations, iterations))
                        saver.save(sess, os.path.join(self.check_points_path, 'check_points'), epoch + latest_training_epochs + 1)
            writer.close()
            print('Time taken for training is {} min'.format((time.time() - training_start_time) / 60))

    @staticmethod
    def generate_image(save_path, image_pages, images_per_row):
        """
        generate images using the latest saved check points and the images will be saved in 'save_path/images/'
        :param save_path: check points that have been saved in 'save_path/check_points/'
        :param image_pages: generated figure pages
        :param images_per_row: the number of generated images per row/column in a figure
        """
        assert images_per_row <= 10, 'too much images in a figure'
        check_points_path = os.path.join(save_path, 'check_points')
        output_image_path = os.path.join(save_path, 'images')
        components.create_folder(output_image_path, False)
        latest_training_epochs_path = tf.train.latest_checkpoint(check_points_path)
        assert latest_training_epochs_path is not None, "no check points found"
        saver = tf.train.import_meta_graph(latest_training_epochs_path + '.meta')
        latest_training_epochs = int(latest_training_epochs_path.split("-")[1])
        shutil.copy(save_path + os.path.sep + "parameters.txt", output_image_path + os.path.sep + "parameters.txt")  # copy parameter text
        with tf.Session() as sess:
            saver.restore(sess, latest_training_epochs_path)
            for i in range(image_pages):
                generated_images = sess.run('generator/output/during_inference:0')  # only generate 100 images each call
                Gan.__save_images(output_image_path, generated_images, images_per_row, latest_training_epochs, i)

    @staticmethod
    def __discriminator_loss(discriminator_output, batch_size, algorithm):
        with tf.name_scope("discriminator_loss"):
            real_output = tf.slice(discriminator_output, begin=[0, 0], size=[batch_size, -1], name='real_output')
            generated_output = tf.slice(discriminator_output, begin=[batch_size, 0], size=[batch_size, -1], name='generated_output')
            if algorithm == 'vanilla':
                real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
                generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)
                return real_loss + generated_loss
            elif algorithm == 'wgan' or algorithm == 'sn-wgan':
                return -tf.reduce_mean(real_output) + tf.reduce_mean(generated_output)
            else:
                raise ValueError('unknown input training algorithm')

    @staticmethod
    def __generator_loss(generated_output, algorithm):
        with tf.name_scope("generator_loss"):
            if algorithm == 'vanilla':
                return tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(generated_output), logits=generated_output)
            elif algorithm == 'wgan' or algorithm == 'sn-wgan':
                return -tf.reduce_mean(generated_output)
            else:
                raise ValueError('unknown input training algorithm')

    @staticmethod
    def __save_images(output_path, predictions, images_per_row, epoch, index=0):
        fig = plt.figure(figsize=(images_per_row, images_per_row))
        fig.suptitle('epoch_{:04d}.png'.format(epoch))
        for i in range(images_per_row ** 2):
            plt.subplot(images_per_row, images_per_row, i + 1)
            if predictions.shape[1] == 1:
                plt.imshow(predictions[i][0] * 127.5 + 127.5, cmap='gray')
                plt.axis('off')
            else:
                image = np.around(predictions[i] * 127.5 + 127.5).astype(int)
                image = np.transpose(image, [2, 1, 0])
                plt.imshow(image)
                plt.axis('off')
        plt.savefig(output_path + os.path.sep + 'epoch_{:04d}_{}.png'.format(epoch, index))
        plt.close(fig)
