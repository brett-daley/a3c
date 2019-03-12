import tensorflow as tf
import numpy as np
from itertools import count
from threading import Thread
from time import time
from utils import *


def calculate_lambda_returns(rewards, values, done, discount, lambd):
    if done:
        values[-1] = 0.0
    lambda_returns = rewards + (discount * values[1:])
    for i in reversed(range(len(rewards) - 1)):
        lambda_returns[i] += (discount * lambd) * (lambda_returns[i+1] - values[i+1])
    return lambda_returns


def calculate_renormalized_lambda_returns(rewards, values, done, discount, lambd):
    assert lambd != 1.0
    if done:
        values[-1] = 0.0
    lambda_returns = rewards + (discount * values[1:])
    N = 1
    for i in reversed(range(len(rewards) - 1)):
        def k(n):
            if n == 0:
                return 1.0
            return sum([lambd**i for i in range(n)])
        N += 1
        lambda_returns[i] = (1. / k(N)) * (lambda_returns[i] + lambd * k(N-1) * (rewards[i] + discount * lambda_returns[i+1]))
    return lambda_returns


def execute(
        make_env,
        policy,
        optimizer,
        discount,
        lambda_pi,
        lambda_ve,
        renormalize,
        entropy_bonus,
        max_sample_length,
        n_actors,
        max_timesteps,
        grad_clip=None,
        log_every_n_steps=100000,
        mov_avg_size=300,
    ):

    training_envs = [make_env() for i in range(n_actors)]
    benchmark_env = make_env()

    input_shape = benchmark_env.observation_space.shape
    input_dtype = benchmark_env.observation_space.dtype
    n_actions   = benchmark_env.action_space.n

    with tf.Session() as session:
        state_ph      = tf.placeholder(input_dtype, [None] + list(input_shape))
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

        def policy(state):
            distr = session.run(action_distr, feed_dict={state_ph: state[None]})[0]
            action = np.random.choice(np.arange(n_actions), p=distr)
            return action

        shared_counter = ThreadsafeCounter(period=log_every_n_steps)
        shared_rewards = ThreadsafeRewards(maxlen=mov_avg_size)
        shared_rewards.extend(benchmark(benchmark_env, policy, n_episodes=mov_avg_size))


        class Actor:
            def __init__(self, env):
                self.env = env
                self.state = None
                self.done = True
                self.thread = None

            def start(self):
                self.thread = Thread(target=self._train)
                self.thread.start()

            def join(self):
                assert self.thread is not None
                self.thread.join()

            def _train(self):
                while not shared_counter.is_expired():
                    states, actions, pi_returns, ve_returns = self._sample()

                    shared_counter.increment(len(states))

                    session.run(train_op, feed_dict={
                        state_ph:     states,
                        action_ph:    actions,
                        pi_return_ph: pi_returns,
                        ve_return_ph: ve_returns,
                    })

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
                    if self.get_total_episodes() > 0:
                        shared_rewards.append(get_episode_rewards(self.env)[-1])

                states.append(state)

                for t in count():
                    if t == max_sample_length or done:
                        break

                    action = policy(state)

                    state, reward, done, _ = self.env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                self.state = state
                self.done = done

                states  = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                values = self._value(states)

                if renormalize:
                    pi_returns = calculate_renormalized_lambda_returns(rewards, values, self.done, discount, lambd=lambda_pi)
                    ve_returns = calculate_renormalized_lambda_returns(rewards, values, self.done, discount, lambd=lambda_ve)
                else:
                    pi_returns = calculate_lambda_returns(rewards, values, self.done, discount, lambd=lambda_pi)
                    ve_returns = calculate_lambda_returns(rewards, values, self.done, discount, lambd=lambda_ve)

                return states[:-1], actions, pi_returns, ve_returns

            def get_total_episodes(self):
                return len(get_episode_rewards(self.env))


        actors = [Actor(training_envs[i]) for i in range(n_actors)]
        timesteps = 0
        best_mean_reward = -float('inf')
        start_time = time()

        for epoch in count():
            print('Epoch', epoch)
            print('Timestep', timesteps)
            print('Realtime {:.3f}'.format(time() - start_time))
            print('Episodes', sum([a.get_total_episodes() for a in actors]))

            mean_reward = shared_rewards.mean()
            std_reward = shared_rewards.std()
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
