import tensorflow as tf
import numpy as np
import itertools
import gym
import threading
import math
import time
import utils
import wrappers


def execute(
        make_env,
        policy,
        optimizer,
        discount,
        lambda_pi,
        lambda_ve,
        entropy_bonus,
        max_sample_length,
        n_actors,
        max_timesteps,
        state_dtype=tf.float32,
        log_every_n_steps=25000,
        grad_clip=None,
    ):

    training_envs = [make_env() for i in range(n_actors)]
    benchmark_env = make_env()

    input_shape = benchmark_env.observation_space.shape
    n_actions   = benchmark_env.action_space.n

    with tf.Session() as session:
        state_ph      = tf.placeholder(state_dtype, [None] + list(input_shape))
        action_ph     = tf.placeholder(tf.int32,    [None])
        pi_return_ph  = tf.placeholder(tf.float32,  [None])
        ve_return_ph  = tf.placeholder(tf.float32,  [None])

        action_distr, value = policy(state_ph, n_actions, scope='policy')

        action_indices = tf.stack([tf.range(tf.size(action_ph)), action_ph], axis=1)
        action_probs = tf.gather_nd(action_distr, action_indices)

        objective = tf.reduce_mean(tf.log(action_probs + 1e-10) * (pi_return_ph - tf.stop_gradient(value)))
        loss      = tf.reduce_mean(tf.square(ve_return_ph - value))
        entropy   = (-entropy_bonus) * tf.reduce_mean(
                        tf.reduce_sum(action_distr * tf.log(action_distr + 1e-10), axis=1)
                    )

        grads_and_vars = optimizer.compute_gradients(loss - (objective + entropy))
        if grad_clip is not None:
            grads_and_vars = [(tf.clip_by_value(g, -grad_clip, +grad_clip), v) for g, v in grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars)

        session.run(tf.global_variables_initializer())

        shared_counter = utils.Counter(period=log_every_n_steps)


        class Actor:
            def __init__(self, env, counter):
                self.env = env
                self.epoch_begin = 0

                self.state   = None
                self.done    = True

                self.counter = counter
                self.thread  = None

            def start(self):
                self.thread = threading.Thread(target=self._train)
                self.thread.start()

            def join(self):
                assert self.thread is not None
                self.thread.join()

            def _train(self):
                while not self.counter.is_expired():
                    states, actions, pi_returns, ve_returns = self._sample()

                    self.counter.increment(len(states))

                    session.run(train_op, feed_dict={
                        state_ph:     states,
                        action_ph:    actions,
                        pi_return_ph: pi_returns,
                        ve_return_ph: ve_returns,
                    })

            def policy(self, state):
                distr = session.run(action_distr, feed_dict={state_ph: state[None]})[0]
                action = np.random.choice(np.arange(n_actions), p=distr)
                return action

            def _value(self, states):
                return session.run(value, feed_dict={state_ph: states})

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

                    action = self.policy(state)

                    state, reward, done, _ = self.env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                self.state = state
                self.done = done

                states  = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)

                pi_returns = self._compute_returns(states, rewards, lambd=lambda_pi)
                ve_returns = self._compute_returns(states, rewards, lambd=lambda_ve)

                return states[:-1], actions, pi_returns, ve_returns

            def _compute_returns(self, states, rewards, lambd=1.0):
                values = self._value(states)
                if self.done:
                    values[-1] = 0.
                returns = rewards + (discount * values[1:])

                for i in reversed(range(len(returns) - 1)):
                    returns[i] += (discount * lambd) * (returns[i+1] - values[i+1])

                return returns

            def _get_episode_rewards(self):
                return utils.get_episode_rewards(self.env)

            def get_total_episodes(self):
                return len(self._get_episode_rewards())

            def get_epoch_rewards(self):
                rewards = self._get_episode_rewards()[self.epoch_begin:]
                self.epoch_begin = self.get_total_episodes()
                return rewards


        actors = [Actor(training_envs[i], shared_counter) for i in range(n_actors)]
        timesteps = 0
        best_mean_reward = -float('inf')
        start_time = time.time()

        for epoch in itertools.count():
            print('Epoch', epoch)
            print('Timestep', timesteps)
            print('Realtime {:.3f}'.format(time.time() - start_time))
            print('Episodes', sum([a.get_total_episodes() for a in actors]))

            if epoch == 0:
                rewards = utils.benchmark(benchmark_env, actors[0].policy, n_episodes=100)
            else:
                rewards = list(itertools.chain.from_iterable(
                              [a.get_epoch_rewards() for a in actors]
                          ))

            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            best_mean_reward = max(mean_reward, best_mean_reward)

            print('Mean reward', mean_reward)
            print('Best mean reward', best_mean_reward)
            print('Standard dev', std_reward)
            print(flush=True)

            if timesteps >= max_timesteps:
                break

            for a in actors:
                a.start()

            for a in actors:
                a.join()

            timesteps += shared_counter.value()
            shared_counter.reset()
