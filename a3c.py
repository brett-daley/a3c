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
        n_actors,
        n_iterations,
        log_every_n_iterations=2000,
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
        entropy = -tf.reduce_sum(log_action_probs * action_probs)

        train_op = optimizer.minimize(-loss1 + loss2 + (0.01 * entropy), var_list=policy_vars)

        session.run(tf.global_variables_initializer())

        def policy_func(state):
            distr = action_distr.eval(feed_dict={state_ph: state[None]})[0]
            action = np.random.choice(np.arange(n_actions), p=distr)
            return action

        def value_func(state):
            return value.eval(feed_dict={state_ph: state[None]})[0]

        def new_actor():
            return utils.Actor(env, discount, policy_func, value_func)

        actors = [new_actor() for i in range(n_actors)]

        for i in range(n_iterations):
            x = np.random.randint(n_actors)
            states, actions, returns = actors[x].sample(t_max=max_sample_length)

            session.run(train_op, feed_dict={
                state_ph:  states,
                action_ph: actions,
                return_ph: returns,
            })

            if i % log_every_n_iterations == 0:
                n_episodes = sum([actors[j].get_n_episodes() for j in range(n_actors)])
                mean_reward = np.mean([actors[j].get_average_reward(n=10) for j in range(n_actors)])

                print('Iteration', i)
                print('Episodes {}'.format(n_episodes))
                print('Mean reward (10 episodes) {}'.format(mean_reward))
                print(flush=True)
