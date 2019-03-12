import gym
import tensorflow as tf
import a3c
import utils
from policies import CartPolePolicy


def make_env(name):
    env = gym.make(name)
    env = gym.wrappers.Monitor(env, 'videos/', force=True, video_callable=lambda e: False)
    env.seed(utils.random_seed())
    return env


def main():
    env_name = 'CartPole-v0'

    utils.seed_all(seed=0)

    optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, use_locking=True)

    a3c.execute(
        lambda: make_env(env_name),
        CartPolePolicy,
        optimizer,
        discount=0.99,
        lambda_pi=1.0,
        lambda_ve=1.0,
        renormalize=False,
        entropy_bonus=0.01,
        max_sample_length=20,
        n_actors=16,
        max_timesteps=1000000,
        log_every_n_steps=10000,
    )


if __name__ == '__main__':
    main()
