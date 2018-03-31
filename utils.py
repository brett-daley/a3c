import tensorflow as tf
import numpy as np


def seed_all(seed=None):
    if seed is None:
        seed = random_seed()

    print('Using {} as random seed'.format(seed))

    tf.set_random_seed(seed)
    np.random.seed(seed)


def random_seed():
    return np.random.randint(65536)


class Counter:
    def __init__(self):
        self._value = 0

    def value(self):
        return self._value

    def increment(self, x):
        self._value += x
