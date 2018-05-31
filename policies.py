import tensorflow as tf
from tensorflow.python.layers.layers import *


def CartPolePolicy(state, n_actions, scope):
    with tf.variable_scope(scope):
        hidden = state
        hidden = dense(hidden, units=64, activation=tf.nn.tanh)
        hidden = dense(hidden, units=64, activation=tf.nn.tanh)

        action_distr = dense(hidden, units=n_actions, activation=tf.nn.softmax)
        value        = dense(hidden, units=1,         activation=None)[:, 0]

    return action_distr, value


def AtariPolicy(state, n_actions, scope):
    state = tf.cast(state, tf.float32) / 255.0

    with tf.variable_scope(scope):
        hidden = state
        hidden = conv2d(hidden, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
        hidden = conv2d(hidden, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
        hidden = conv2d(hidden, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)

        hidden = flatten(hidden)
        hidden = dense(hidden, units=512, activation=tf.nn.relu)

        action_distr = dense(hidden, units=n_actions, activation=tf.nn.softmax)
        value        = dense(hidden, units=1,         activation=None)[:, 0]

    return action_distr, value
