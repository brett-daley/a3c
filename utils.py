import tensorflow as tf
import numpy as np
import threading


def seed_all(seed=None):
    if seed is None:
        seed = random_seed()

    print('Using {} as random seed'.format(seed))

    tf.set_random_seed(seed)
    np.random.seed(seed)


def random_seed():
    return np.random.randint(65536)


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
