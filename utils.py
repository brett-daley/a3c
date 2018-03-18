import tensorflow as tf
import numpy as np


def seed_all(env, seed=None):
    if seed is None:
        seed = np.random.randint(65536)

    print('Using {} as random seed'.format(seed))

    env.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
