import tensorflow as tf
import tensorflow.contrib.layers as layers


def CartPolePolicy(state, n_actions, scope):
    with tf.variable_scope(scope):
        hidden = layers.fully_connected(state,  num_outputs=1024,            activation_fn=tf.nn.relu)
        hidden = layers.fully_connected(hidden, num_outputs=(n_actions + 1), activation_fn=None)

        action_distr = tf.nn.softmax(hidden[:, :-1])
        value = hidden[:, -1]

        return action_distr, value
