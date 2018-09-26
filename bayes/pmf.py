'''
Probabilisitic Matrix Factorization using Tensorflow
'''

import numpy as np
import tensorflow as tf
from scipy import sparse

NUM_ITEMS = 20000
NUM_USERS = 1000
RANK = 20
NUM_EPOCHS = 10000


def build_toy_dataset(noise_std=0.1):
    U = sparse.random(RANK, NUM_USERS, 0.05)
    V = sparse.random(RANK, NUM_ITEMS, 0.05)
    R = U.transpose() * V
    nonzero_rows, nonzero_cols, nonzero_vals = sparse.find(R)
    return U, V, R, nonzero_rows, nonzero_cols, nonzero_vals


U_true, V_true, R_true, nonzero_rows, nonzero_cols, nonzero_vals = \
    build_toy_dataset()

index = np.vstack([nonzero_rows, nonzero_cols]).T
nonzero_R_true = R_true[nonzero_rows, nonzero_cols]

U = tf.get_variable('U', [RANK, NUM_USERS], tf.float32,
                    tf.truncated_normal_initializer(mean=0.0, stddev=0.2))
V = tf.get_variable('V', [RANK, NUM_ITEMS], tf.float32,
                    tf.truncated_normal_initializer(mean=0.0, stddev=0.2))
R = tf.matmul(tf.transpose(U), V)

lambda_U = 0.1
lambda_V = 0.1
error = tf.reduce_sum((nonzero_R_true - tf.gather_nd(R, index))**2)
regularization_U = lambda_U * tf.reduce_sum(tf.norm(U, axis=1))
regularization_V = lambda_V * tf.reduce_sum(tf.norm(V, axis=1))

loss = error + regularization_U + regularization_V
optim = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_EPOCHS):
        loss_, _ = sess.run([loss, optim])
        print(loss_)
