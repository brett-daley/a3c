import gym
import tensorflow as tf
import argparse

import a3c
import utils
import policies
import wrappers
from policies import AtariPolicy


def make_atari_env(name, history_len):
    from gym.envs.atari.atari_env import AtariEnv
    from gym.wrappers.monitor import Monitor
    env = AtariEnv(game=name, frameskip=4, obs_type='image')
    env = Monitor(env, 'videos/', force=True, video_callable=lambda e: False)
    env = wrappers.wrap_deepmind(env)
    env = wrappers.HistoryWrapper(env, history_len)
    env.seed(utils.random_seed())
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',         type=str,   default='pong')
    parser.add_argument('--lambd',       type=float, default=1.0)
    parser.add_argument('--history-len', type=int,   default=4)
    parser.add_argument('--seed',        type=int,   default=0)
    parser.add_argument('--renorm',      action='store_true')
    args = parser.parse_args()

    utils.seed_all(seed=args.seed)

    optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, epsilon=1e-4, use_locking=True)

    a3c.execute(
        lambda: make_atari_env(args.env, args.history_len),
        AtariPolicy,
        optimizer,
        discount=0.99,
        lambd=args.lambd,
        renormalize=args.renorm,
        entropy_bonus=0.01,
        max_sample_length=10,
        n_actors=16,
        max_timesteps=40000000,
        grad_clip=40.,
        log_every_n_steps=1000,
    )


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()
