# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : gan.py
# @Time    : 2018/11/12 20:49
# @Author  : LU Tianle

"""
training algorithms of vanilla gan
"""
import os
import time

import imageio
import numpy as np
import tensorflow as tf
import components
import plot


class Gan:
    def __init__(self, generator, discriminator, noise_dim, save_path):
        """
        construct the gan nets
        :param generator: the object of generator networks
        :param discriminator: the networks of discriminator networks
         :param noise_dim: noise dim for generation
        :param save_path: check points and output image will be saved in folders in this path
        """
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim
        self.save_path = save_path
        self.check_points_path = os.path.join(save_path, 'check_points')
        self.output_image_path = os.path.join(save_path, 'images_during_training')
        noise = tf.random_normal(shape=(200, self.noise_dim), name='noise_for_inference')
        generated_images = self.generator(noise, training=False, name='inference')

    def train(self, dataset, batch_size, epochs, algorithm,
              discriminator_training_loop, discriminator_optimizer, generator_optimizer,
              images_per_row, continue_training=False):
        """
        start training and save models
        :param dataset: training dataset
        :param batch_size: mini batch size
        :param epochs: total training epochs
        :param algorithm: 'vanilla or wgan'
        :param discriminator_training_loop: update generator once while updating discriminator more times
        :param discriminator_optimizer: tf.train.Optimizer
        :param generator_optimizer: tf.train.Optimizer
        :param images_per_row: the number of generated images per row/column in a figure every epochs
        :param continue_training: continue training using the latest check points in the self.save_path
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        components.create_folder(self.save_path, continue_training)
        components.create_folder(self.check_points_path, continue_training)
        components.create_folder(self.output_image_path, continue_training)
        # saved training iterations
        saved_iterations = tf.get_variable(name='saved_iterations', dtype=tf.int32, initializer=0, trainable=False)
        # create training dataset
        with tf.name_scope('dataset'):
            dataset = dataset.batch(batch_size, drop_remainder=True)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            iterator_initialize = iterator.make_initializer(dataset)
        # compute loss
        noise_for_training = tf.random_normal([batch_size, self.noise_dim], name='noise_for_training')
        generated_images = self.generator(noise_for_training, training=True, name='training')
        disc_real = self.discriminator(iterator.get_next(), training=True, name='disc_real')
        disc_fake = self.discriminator(generated_images, training=True, name='disc_fake')
        discriminator_loss = Gan.__discriminator_loss(disc_real, disc_fake, algorithm)
        generator_loss = Gan.__generator_loss(disc_fake, algorithm)
        # optimization
        with tf.control_dependencies([op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if "discriminator" in op.name]):
            train_discriminator = discriminator_optimizer.minimize(discriminator_loss, var_list=self.discriminator.var_list, name='discriminator_optimizer')
        with tf.control_dependencies([op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if "generator" in op.name]):
            train_generator = generator_optimizer.minimize(generator_loss, var_list=self.generator.var_list, name='generator_optimizer')
        # weight clipping in wgan and sn-wgan
        if algorithm == 'wgan':
            clip = ([var.assign(tf.clip_by_value(var, -0.01, 0.01, name='clip/' + var.op.name), name='clip/' + var.op.name)
                     for var in self.discriminator.var_list])
            clip = tf.group(*clip)
        elif algorithm == 'sn-wgan':
            clip = ([var.assign(tf.clip_by_value(var, -0.01, 0.01, name='clip/' + var.op.name), name='clip/' + var.op.name)
                     for var in self.discriminator.var_list if 'gamma' in var.name])
            clip = tf.group(*clip)
        else:
            clip = None
        # images will be generated after epochs using the same noise
        noise_for_generation = tf.constant(np.random.normal(size=(images_per_row ** 2, self.noise_dim)).astype('float32'), name='noise_for_generation')
        generate_images_each_epoch = self.generator(noise_for_generation, training=False, name='inference_each_epoch')
        saver = tf.train.Saver(max_to_keep=3)
        with tf.Session(config=config) as sess:
            if continue_training:  # continue training from the latest check point
                latest_training_epochs_path = tf.train.latest_checkpoint(self.check_points_path)
                assert latest_training_epochs_path is not None, "no check points found"
                latest_training_epochs = int(latest_training_epochs_path.split("-")[-1])  # load training epochs
                saver.restore(sess, latest_training_epochs_path)
                iterations = sess.run('saved_iterations:0')  # load training iterations
            else:  # start a initial training
                latest_training_epochs = 0
                iterations = 1  # training mini-batch iterations
                sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(self.save_path, sess.graph)
            training_start_time = time.time()
            for epoch in range(epochs):  # training loop
                epoch_start_time = time.time()
                sess.run(iterator_initialize)
                try:
                    while True:
                        save_discriminator_loss = 0
                        for i in range(discriminator_training_loop):
                            if i == discriminator_training_loop - 1:
                                save_discriminator_loss, _ = sess.run([discriminator_loss, train_discriminator])
                            else:
                                sess.run([train_discriminator])
                            if clip is not None:
                                sess.run(clip)
                        Gan.__save_summaries(algorithm, save_discriminator_loss)
                        sess.run(train_generator)
                        iterations += 1
                except tf.errors.OutOfRangeError:
                    pass
                plot.flush(self.save_path)
                print('Time taken for epoch {} is {} sec'.format(epoch + latest_training_epochs + 1, time.time() - epoch_start_time))
                # saving generated images after each epoch
                predictions = sess.run(generate_images_each_epoch)
                Gan.__save_images(self.output_image_path, predictions, images_per_row, iterations - 1)
                if (epoch + 1) % 5 == 0:
                    sess.run(tf.assign(saved_iterations, iterations - 1))
                    saver.save(sess, os.path.join(self.check_points_path, 'check_points'), iterations - 1)
            print('Time taken for training is {} min'.format((time.time() - training_start_time) / 60))
            writer.close()

    def generate_image(self, save_path, image_pages, images_per_row):
        """
        generate images using the latest saved check points and the images will be saved in 'save_path/images/'
        :param save_path: check points that have been saved in 'save_path/check_points/'
        :param image_pages: generated figure pages
        :param images_per_row: the number of generated images per row/column in a figure
        """
        check_points_path = os.path.join(save_path, 'check_points')
        output_image_path = os.path.join(save_path, 'images')
        components.create_folder(output_image_path, False)
        latest_checkpoint = tf.train.latest_checkpoint(check_points_path)
        assert latest_checkpoint is not None, "no check points found"
        saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
        for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'):
            print(x.op.name)
        with tf.Session() as sess:
            saver.restore(sess, latest_checkpoint)
            iterations = sess.run('saved_iterations:0')
            for i in range(image_pages):
                generated_images = sess.run('inference:0')
                Gan.__save_images(output_image_path, generated_images, images_per_row, iterations, i)

    @staticmethod
    def __save_summaries(algorithm, discriminator_loss):
        with tf.name_scope('summary'):
            if algorithm == 'vanilla':
                plot.plot('JS_divergence', -0.5 * discriminator_loss + 1)
            elif algorithm == 'wgan' or algorithm == 'sn-wgan':
                plot.plot('Wasserstein_distance', -discriminator_loss)
            else:
                raise ValueError('unknown input training algorithm')

    @staticmethod
    def __discriminator_loss(real_output, generated_output, algorithm):
        with tf.name_scope("discriminator_loss"):
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
    def __save_images(output_path, predictions, images_per_row, iteration, index=0):
        [_, channel, height, width] = predictions.shape
        if channel == 1:
            images = np.around(predictions * 127.5 + 127.5).astype('uint8')
            fig = np.zeros([height * images_per_row, width * images_per_row], dtype='uint8')
            for i in range(images_per_row):
                for j in range(images_per_row):
                    fig[i * height:(i + 1) * height, j * width:(j + 1) * width] = images[i * images_per_row + j][0]
        else:
            fig = np.zeros([height * images_per_row, width * images_per_row, 3], dtype='uint8')
            images = np.around(np.transpose(predictions, [0, 2, 3, 1]) * 127.5 + 127.5).astype('uint8')
            for i in range(images_per_row):
                for j in range(images_per_row):
                    fig[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = images[i * images_per_row + j]
        imageio.imwrite(os.path.join(output_path, 'generator iteration: {}_{}.png'.format(iteration, index)), fig)
