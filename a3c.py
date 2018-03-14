import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import utils


def policy(state, n_actions, scope):
    with tf.variable_scope(scope):
        hidden = layers.fully_connected(state,  num_outputs=1024,            activation_fn=tf.nn.relu)
        hidden = layers.fully_connected(hidden, num_outputs=(n_actions + 1), activation_fn=None)

        action_distr = tf.nn.softmax(hidden[:, :-1])
        value = hidden[:, -1]

        return action_distr, value


def execute(
        env,
        optimizer,
        discount,
        max_sample_length,
        n_iterations,
        log_every_n_iterations=100,
    ):

    input_size, = env.observation_space.shape
    n_actions   = env.action_space.n

    with tf.Session() as session:
        state_ph      = tf.placeholder(tf.float32, [None, input_size])
        action_ph     = tf.placeholder(tf.int32,   [None])
        return_ph     = tf.placeholder(tf.float32, [None])

        action_distr, value = policy(state_ph, n_actions, scope='policy')
        policy_vars = tf.trainable_variables(scope='policy')

        action_indices = tf.stack([tf.range(tf.size(action_ph)), action_ph], axis=1)
        action_probs = tf.gather_nd(action_distr, action_indices)
        log_action_probs = tf.log(action_probs)

        advantages = return_ph - value

        loss1 = tf.reduce_sum(log_action_probs * advantages)
        loss2 = tf.reduce_sum(tf.square(advantages))

        train_op = optimizer.minimize(-loss1 + loss2, var_list=policy_vars)

        session.run(tf.global_variables_initializer())

        sampler = utils.Sampler(
            env,
            discount,
            policy_func=lambda x: action_distr.eval(feed_dict={state_ph: x[None]})[0],
            value_func=lambda x: value.eval(feed_dict={state_ph: x[None]})[0],
        )

        for i in range(n_iterations):
            states, actions, returns = sampler.sample(t_max=max_sample_length)

            feed_dict = {
                state_ph:  states,
                action_ph: actions,
                return_ph: returns,
            }

            session.run(train_op, feed_dict=feed_dict)

            if i % log_every_n_iterations == 0:
                episode_rewards = env.get_episode_rewards()

                print('Iteration', i)
                print('Episodes {}'.format(len(episode_rewards)))
                print('Mean reward (100 episodes) {}'.format(np.mean(episode_rewards[:-100])))
                print(flush=True)
