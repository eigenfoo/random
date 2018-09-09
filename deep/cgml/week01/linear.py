#!/bin/python3.6

import numpy as np
import tensorflow as tf

from tqdm import tqdm

NUM_FEATURES = 4
BATCH_SIZE = 32
NUM_BATCHES = 300

class Data(object):
    def __init__(self):
        num_samp = 50
        sigma = 0.1
        np.random.seed(31415)

        # We're going to learn these paramters 
        w = np.atleast_2d([4, 3, 4, 2]).transpose()
        b = 2

        self.index = np.arange(num_samp)        
        self.x = np.random.uniform(size=(num_samp, 4))
        self. y = self.x @ w + b + sigma * np.random.normal()
        
    def get_batch(self):
        choices = np.random.choice(self.index, size=BATCH_SIZE)

        return self.x[choices], self.y[choices].flatten()


def f(x):
    w = tf.get_variable('w', [NUM_FEATURES, 1], tf.float32,
                        tf.random_normal_initializer())
    b = tf.get_variable('b', [], tf.float32, tf.zeros_initializer())

    return tf.squeeze(tf.matmul(x, w) + b)


x = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_FEATURES])
y = tf.placeholder(tf.float32, [BATCH_SIZE])
y_hat = f(x)

loss = tf.reduce_mean(tf.pow(y_hat - y, 2))
optim = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

data = Data()

for _ in tqdm(range(0, NUM_BATCHES)):
    x_np, y_np = data.get_batch()
    loss_np, _ = sess.run([loss, optim], feed_dict={x: x_np, y: y_np})

print("Parameter estimates:")
for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(
        var.name.rstrip(":0"),
        np.array_str(np.array(sess.run(var)).flatten(), precision=3))
