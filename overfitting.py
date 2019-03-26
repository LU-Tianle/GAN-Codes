# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : overfitting.py 
# @Time    : 2019/3/25 11:31
# @Author  : LU Tianle

"""
test whether the model has been overfitted use the latest saved check points
the generated images will be saved in SAVE_PATH//overfitting_test_images
"""
import os

import imageio
import numpy as np
import tensorflow as tf

SAVE_PATH = './mnist//Inception-trans Nets vanilla'
Noise_DIM = 128
ROWS = 5
COLUMNS = 10

k = np.random.uniform(size=[COLUMNS - 2]).astype('float32')
check_points_path = os.path.join(SAVE_PATH, 'check_points')
output_image_path = os.path.join(SAVE_PATH, 'overfitting_test_images')
if not os.path.exists(output_image_path):
    os.makedirs(output_image_path)
latest_checkpoint = tf.train.latest_checkpoint(check_points_path)
assert latest_checkpoint is not None, "no check points found"
saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')

noise = None
for i in range(ROWS):
    noise_first = np.random.randn(1, Noise_DIM).astype('float32')
    noise_last = noise_first * 2 + 1
    if noise is None:
        noise = noise_first
    else:
        noise = np.concatenate([noise, noise_first], axis=0)
    for j in range(COLUMNS - 2):
        interpolation_image = k[j] * noise_first + (1 - k[j]) * noise_last
        noise = np.concatenate([noise, interpolation_image], axis=0)
    noise = np.concatenate([noise, noise_last], axis=0)

with tf.Session() as sess:
    saver.restore(sess, latest_checkpoint)
    iterations = sess.run('saved_iterations:0')
    generated_images = sess.run('generator/output_layer/tanh/during_inference:0',
                                feed_dict={"noise_for_inference:0": noise})

[_, height, width, channel] = generated_images.shape
if channel == 1:
    images = np.around(generated_images * 127.5 + 127.5).astype('uint8')
    fig = np.zeros([height * ROWS, width * COLUMNS], dtype='uint8')
    for i in range(ROWS):
        for j in range(COLUMNS):
            fig[i * height:(i + 1) * height, j * width:(j + 1) * width] = images[i * COLUMNS + j][:, :, 0]
else:
    fig = np.zeros([height * ROWS, width * COLUMNS, 3], dtype='uint8')
    images = np.around(generated_images * 127.5 + 127.5).astype('uint8')
    for i in range(ROWS):
        for j in range(COLUMNS):
            fig[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = images[i * COLUMNS + j]
imageio.imwrite(os.path.join(output_image_path, 'generator_iteration_{}.png'.format(iterations)), fig)
