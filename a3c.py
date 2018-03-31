import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import itertools
import copy
import gym
import threading
import math
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
        objective_optimizer,
        loss_optimizer,
        discount,
        entropy_bonus,
        max_sample_length,
        n_actors,
        max_timesteps,
        log_every_n_steps=10000,
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
        log_action_probs = tf.log(action_probs + 1e-30) / tf.log(2.)

        objective = tf.reduce_sum(log_action_probs * (return_ph - tf.stop_gradient(value)))
        loss      = tf.reduce_sum(tf.square(return_ph - value))
        entropy   = entropy_bonus * (-tf.reduce_sum(log_action_probs * action_probs))

        grads_and_vars     = objective_optimizer.compute_gradients(-(objective + entropy), var_list=policy_vars)
        objective_train_op = objective_optimizer.apply_gradients(grads_and_vars)

        grads_and_vars = loss_optimizer.compute_gradients(loss, var_list=policy_vars)
        loss_train_op  = loss_optimizer.apply_gradients(grads_and_vars)

        train_op = tf.group(*[objective_train_op, loss_train_op])

        session.run(tf.global_variables_initializer())

        shared_counter = utils.Counter()


        class Actor:
            def __init__(self, counter):
                self.env     = copy.deepcopy(env)
                self.env     = gym.wrappers.Monitor(self.env, 'videos/', force=True, video_callable=lambda e: False)

                self.state   = None
                self.done    = True

                self.counter = counter
                self.thread  = threading.Thread(target=self._train)

            def start(self):
                self.thread.start()

            def join(self):
                self.thread.join()

            def _train(self):
                while self.counter.value() < max_timesteps:
                    states, actions, returns = self._sample()

                    self.counter.increment(len(states))

                    session.run(train_op, feed_dict={
                        state_ph:  states,
                        action_ph: actions,
                        return_ph: returns,
                    })

            def _policy(self, state):
                distr = session.run(action_distr, feed_dict={state_ph: state[None]})[0]
                action = np.random.choice(np.arange(n_actions), p=distr)
                return action

            def _value(self, state):
                return session.run(value, feed_dict={state_ph: state[None]})[0]

            def _sample(self):
                states  = []
                actions = []
                rewards = []

                state = self.state
                done = self.done

                if done:
                    state = self.env.reset()
                    done = False

                states.append(state)

                for t in itertools.count():
                    if t == max_sample_length or done:
                        break

                    action = self._policy(state)

                    state, reward, done, _ = self.env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                self.state = state
                self.done = done

                return np.array(states[:-1]), np.array(actions), self._compute_returns(rewards)

            def _compute_returns(self, rewards):
                last_value = 0. if self.done else self._value(self.state)
                values = last_value * np.array([discount**(i+1) for i in reversed(range(len(rewards)))])

                for i in reversed(range(len(rewards) - 1)):
                    rewards[i] += discount * rewards[i+1]

                return (rewards + values)

            def get_n_episodes(self):
                return len(self.env.get_episode_rewards())

            def get_average_reward(self, n):
                return np.mean(self.env.get_episode_rewards()[-n:])


        actors = [Actor(shared_counter) for i in range(n_actors)]
        for a in actors:
            a.start()

        while True:
            t = shared_counter.value()
            # TODO: need a more elegant way to do this
            import time
            time.sleep(0.01)
            T = shared_counter.value()

            if (T // log_every_n_steps) > (t // log_every_n_steps):
                window_size = math.ceil(100. / n_actors)
                n_episodes = sum([a.get_n_episodes() for a in actors])
                mean_reward = np.mean([a.get_average_reward(window_size) for a in actors])

                print('Timesteps', t)
                print('Episodes', n_episodes)
                print('Mean reward ({} episodes/thread) {:.3f}'.format(window_size, mean_reward))
                print(flush=True)

            if T >= max_timesteps:
                break

        for a in actors:
            a.join()
