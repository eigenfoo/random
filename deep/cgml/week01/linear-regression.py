'''
ECE 471, Selected Topics in Machine Learning – Assignment 1

Submit by Sept. 12, 10pm

**tldr:** Perform linear regression of a noisy sinewave using a set of
gaussian basis functions with learned location and scale parameters. Model
parameters are learned with stochastic gradient descent. Use of automatic
differentiation is required. Hint: note your limits!
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

sns.set_style('whitegrid')

NUM_DATAPOINTS = 50
NUM_FEATURES = 4
BATCH_SIZE = 50  # FIXME rn batch size must be 50... oops
NUM_BATCHES = 100

sigma = 0.1
np.random.seed(31415)
x_np = np.random.uniform(size=NUM_DATAPOINTS)
y_np = np.sin(2*np.pi*x_np) \
       + np.random.normal(scale=sigma, size=NUM_DATAPOINTS)

# Must be column vector for tf broadcasting
x_np = np.atleast_2d(x_np).T
y_np = np.atleast_2d(y_np).T


def f(x):
    mu = tf.get_variable('mu', [NUM_FEATURES, 1], tf.float32,
                         tf.random_normal_initializer())
    sigma = tf.get_variable('sigma', [NUM_FEATURES, 1], tf.float32,
                            tf.ones_initializer())

    # Tensorflow broadcasting
    phi_x = tf.exp(-tf.div(tf.square(x - tf.transpose(mu)),
                           tf.square(tf.transpose(sigma))))

    w = tf.get_variable('w', [NUM_FEATURES, 1], tf.float32,
                        tf.random_normal_initializer())
    b = tf.get_variable('b', [], tf.float32, tf.zeros_initializer())

    return tf.squeeze(tf.matmul(phi_x, w) + b)


x = tf.placeholder(tf.float32, [BATCH_SIZE, 1])
y = tf.placeholder(tf.float32, [BATCH_SIZE, 1])
y_hat = f(x)

loss = tf.reduce_mean(tf.square(y_hat - y))
optim = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for _ in tqdm(range(NUM_BATCHES)):
    # x_np, y_np = get_batch()
    loss_np, _ = sess.run([loss, optim], feed_dict={x: x_np, y: y_np})
    print(loss_np)

params = {}
print("Parameter estimates:")
for var in tf.trainable_variables():
    name = var.name.rstrip(":0")
    value = np.array(sess.run(var)).flatten()
    params[name] = value
    print(name, np.array_str(value, precision=3))


def weighted_bases(x, i):
    ''' Helper function for plotting '''
    return params['w'][i] * np.exp(-(x - params['mu'][i])**2 / params['sigma'][i]**2)


def phi(x, i):
    ''' Helper function for plotting '''
    return np.exp(-(x - params['mu'][i])**2 / params['sigma'][i]**2)


fig, axarr = plt.subplots(ncols=2, nrows=1, figsize=[18, 4])

# First plot
axarr[0].scatter(x_np, y_np, color='g')

x_fit = np.linspace(0, 1, 200)
y_noiseless = np.sin(2*np.pi*x_fit)
axarr[0].plot(x_fit, y_noiseless, color='b')

y_learned = np.vstack([weighted_bases(x_fit, i) for i in range(NUM_FEATURES)]).sum(axis=0) + params['b']
axarr[0].plot(x_fit, y_learned, 'r--')

axarr[0].set_title('Fit')
axarr[0].set_xlabel('x')
axarr[0].set_ylabel('y')

# Second plot
x_bases = np.linspace(-4, 4, 200)

for i in range(NUM_FEATURES):
    axarr[1].plot(x_bases, phi(x_bases, i))

axarr[1].set_title('Bases for Fit')
axarr[1].set_xlabel('x')
axarr[1].set_ylabel('y')

sns.despine()
plt.show()
