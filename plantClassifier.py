#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2
import json
import os
import time
import re

from flask import Flask, Response, request

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

def inference(images_placeholder, keep_prob):
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(images_placeholder, [-1, 28, 28, 3])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
keep_prob = tf.placeholder("float")

logits = inference(images_placeholder, keep_prob)
sess = tf.InteractiveSession()

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
saver.restore(sess, "model.ckpt")

def plant_detect(filename):
    test_image = []
    img = cv2.imread(filename)
    img = cv2.resize(img, (28, 28))
    test_image.append(img.flatten().astype(np.float32)/255.0)
    test_image = np.asarray(test_image)

    for i in range(len(test_image)):
        pred = np.argmax(logits.eval(feed_dict={
            images_placeholder: [test_image[i]],
            keep_prob: 1.0 })[0])
        print "result:"
        print pred
        return str(pred)

def get_dict():
    dict = {}
    n = 0
    for line in open('convert_list.txt', 'r'):
         dict[str(n)] = re.sub('\n','',line)
         n = n + 1
    return dict

print "test"
app = Flask(__name__, static_url_path='', static_folder='public')
app.add_url_rule('/', 'root', lambda: app.send_static_file('index.html'))

@app.route('/plant', methods=['GET', 'POST'])
def comments_handler():
        ret = plant_detect("target.jpg")
        return Response(ret, mimetype='text/plain', headers={'Cache-Control': 'no-cache', 'Access-Control-Allow-Origin': '*'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        app.logger.info("Uploaded filename is "+file.filename)
        file.save(os.path.join('image_tmp', 'target.jpg'))
        dict = get_dict()
        ret = plant_detect("image_tmp/target.jpg")
        print dict,ret
        res = "this plant is " + dict[ret] + "<br><a href='/index.html'>return</a>"
        return Response(res, mimetype='text/html', headers={'Cache-Control': 'no-cache', 'Access-Control-Allow-Origin': '*'})


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT",3000)))
