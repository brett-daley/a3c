import tensorflow as tf
import tensorflow.contrib.layers as layers


def CartPolePolicy(state, cell, rnn_state, n_actions, scope):
    with tf.variable_scope(scope):
        hidden = state
        hidden = layers.fully_connected(hidden, num_outputs=1024,            activation_fn=tf.nn.relu)
        hidden = layers.fully_connected(hidden, num_outputs=(n_actions + 1), activation_fn=None)

        action_distr = tf.nn.softmax(hidden[:, :-1])
        value = hidden[:, -1]

        return action_distr, value, None


def AtariPolicy(state, cell, rnn_state, n_actions, scope):
    print('Feedforward', state.shape)
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

    return action_distr, value, None


def AtariRecurrentPolicy(state, cell, rnn_state, n_actions, scope):
    print('Recurrent', state.shape)
    state = tf.cast(state, tf.float32) / 255.0

    with tf.variable_scope(scope):
        hidden = state
        hidden = layers.convolution2d(hidden, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
        hidden = layers.convolution2d(hidden, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        hidden = layers.convolution2d(hidden, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

        hidden = layers.flatten(hidden)
        hidden = tf.expand_dims(hidden, axis=1)

        hidden, new_rnn_state = tf.nn.dynamic_rnn(cell, inputs=hidden, initial_state=rnn_state, dtype=tf.float32, time_major=True)

        hidden = hidden[:, 0]
        hidden = layers.fully_connected(hidden, num_outputs=(n_actions + 1), activation_fn=None)

        action_distr = tf.nn.softmax(hidden[:, :-1])
        value = hidden[:, -1]

    return action_distr, value, new_rnn_state
