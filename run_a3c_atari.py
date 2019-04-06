import gym
import tensorflow as tf
import argparse

import a3c
import utils
import policies
from wrappers import monitor, wrap_deepmind, HistoryWrapper
from policies import AtariPolicy


def make_atari_env(name, history_len):
    from gym.envs.atari.atari_env import AtariEnv
    env = AtariEnv(game=name, frameskip=4, obs_type='image')
    env = monitor(env, name)
    env = wrap_deepmind(env)
    env = HistoryWrapper(env, history_len)
    env.seed(utils.random_seed())
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',         type=str,   default='pong')
    parser.add_argument('--return-type', type=str,   default='lambda-1.0')
    parser.add_argument('--history-len', type=int,   default=4)
    parser.add_argument('--seed',        type=int,   default=0)
    args = parser.parse_args()

    if 'renorm-lambda-' in args.return_type:
        lambd = float( args.return_type.strip('renorm-lambda-') )
        renorm = True
    elif 'lambda-' in args.return_type:
        lambd = float( args.return_type.strip('lambda-') )
        renorm = False
    else:
        raise ValueError('Unrecognized return type')

    utils.seed_all(seed=args.seed)

    optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, epsilon=1e-4, use_locking=True)

    a3c.execute(
        lambda: make_atari_env(args.env, args.history_len),
        AtariPolicy,
        optimizer,
        discount=0.99,
        lambd=lambd,
        renormalize=renorm,
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
