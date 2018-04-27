import tensorflow as tf
import tensorflow.contrib.layers as layers


def CartPolePolicy(state, n_actions, scope):
    with tf.variable_scope(scope):
        hidden = state
        hidden = layers.fully_connected(hidden, num_outputs=1024,            activation_fn=tf.nn.relu)
        hidden = layers.fully_connected(hidden, num_outputs=(n_actions + 1), activation_fn=None)

        action_distr = tf.nn.softmax(hidden[:, :-1])
        value = hidden[:, -1]

        return action_distr, value


def AtariPolicy(state, n_actions, scope):
    state = tf.cast(state, tf.float32) / 255.0

    with tf.variable_scope(scope):
        hidden = state
        hidden = layers.convolution2d(hidden, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
        hidden = layers.convolution2d(hidden, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        hidden = layers.convolution2d(hidden, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

        hidden = layers.flatten(hidden)
        hidden = layers.fully_connected(hidden, num_outputs=512,             activation_fn=tf.nn.relu)
        hidden = layers.fully_connected(hidden, num_outputs=(n_actions + 1), activation_fn=None)

        action_distr = tf.nn.softmax(hidden[:, :-1])
        value = hidden[:, -1]

        return action_distr, value
