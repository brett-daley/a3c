import tensorflow as tf
import numpy as np
import threading
import gym


def seed_all(seed=None):
    if seed is None:
        seed = random_seed()

    print('Using {} as random seed'.format(seed))

    tf.set_random_seed(seed)
    np.random.seed(seed)


def random_seed():
    return np.random.randint(65536)


def get_episode_rewards(env):
    if isinstance(env, gym.wrappers.Monitor):
        return env.get_episode_rewards()
    elif hasattr(env, 'env'):
        return get_episode_rewards(env.env)
    raise ValueError('No Monitor wrapper around env')


def benchmark(env, policy, n_episodes):
    for i in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            action = policy(state)
            state, _, done, _ = env.step(action)

    return get_episode_rewards(env)[-n_episodes:]


class Counter:
    def __init__(self, period):
        self._period = period
        self._value = 0
        self._expiration = period

        self._lock = threading.Lock()

    def value(self):
        return self._value

    def is_expired(self):
        return (self._value >= self._expiration)

    def increment(self, x):
        self._lock.acquire()
        self._value += x
        self._lock.release()

    def reset(self):
        self._expiration = self._period - (self._value - self._expiration)
        self._value = 0
