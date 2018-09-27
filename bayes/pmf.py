'''
Probabilisitic Matrix Factorization using Tensorflow
'''

import numpy as np
import tensorflow as tf
from scipy import sparse

np.random.seed(1618)

NUM_ITEMS = 5000
NUM_USERS = 100
RANK = 10
NUM_EPOCHS = 50


def build_toy_dataset(noise_std=0.1):
    U = sparse.random(RANK, NUM_USERS, 0.05)
    V = sparse.random(RANK, NUM_ITEMS, 0.05)
    R = U.transpose() * V
    nonzero_rows, nonzero_cols, nonzero_vals = sparse.find(R)
    return U, V, R, nonzero_rows, nonzero_cols, nonzero_vals


def variable_summaries(var, name):
    """ Attach summaries to a Tensor (for TensorBoard visualization). """
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        minimum = tf.reduce_min(var)
        maximum = tf.reduce_max(var)

        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', maximum)
        tf.summary.scalar('min', minimum)
        tf.summary.histogram('histogram', var)


U_true, V_true, R_true, nonzero_rows, nonzero_cols, nonzero_vals = \
    build_toy_dataset()

index = np.vstack([nonzero_rows, nonzero_cols]).T
nonzero_R_true = R_true[nonzero_rows, nonzero_cols]

sess = tf.Session()

with tf.name_scope('matrices'):
    U = tf.get_variable('U', [RANK, NUM_USERS], tf.float32,
                        tf.truncated_normal_initializer(mean=0.0, stddev=0.2))
    V = tf.get_variable('V', [RANK, NUM_ITEMS], tf.float32,
                        tf.truncated_normal_initializer(mean=0.0, stddev=0.2))
    R = tf.matmul(tf.transpose(U), V)

variable_summaries(U, name='U')
variable_summaries(V, name='V')
variable_summaries(R, name='R')

with tf.name_scope('loss'):
    lambda_U = 0.1
    lambda_V = 0.1
    error = tf.reduce_sum((nonzero_R_true - tf.gather_nd(R, index))**2)
    regularization_U = lambda_U * tf.reduce_sum(tf.norm(U, axis=1))
    regularization_V = lambda_V * tf.reduce_sum(tf.norm(V, axis=1))
    loss = error + regularization_U + regularization_V

    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer().minimize(loss)

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(NUM_EPOCHS):
    _, summary_, loss_ = sess.run([train_step, merged_summary, loss])
    writer.add_summary(summary_, i)

    if i % 1 == 0:
        print('Epoch {}:'.format(i), loss_)

writer.close()
