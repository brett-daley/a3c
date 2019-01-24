import gym
import tensorflow as tf
import argparse

import a3c
import utils
import policies
import wrappers


def make_atari_env(name):
    from gym.envs.atari.atari_env import AtariEnv
    env = AtariEnv(game=name, frameskip=4, obs_type='image')
    env = wrappers.wrap_deepmind(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',         type=str,   default='pong')
    parser.add_argument('--lambda-pi',   type=float, default=1.0)
    parser.add_argument('--lambda-ve',   type=float, default=1.0)
    parser.add_argument('--history-len', type=int,   default=4)
    parser.add_argument('--seed',        type=int,   default=0)
    args = parser.parse_args()

    utils.seed_all(seed=args.seed)

    policy = policies.AtariPolicy

    optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, epsilon=1e-4, use_locking=True)

    a3c.execute(
        lambda: make_atari_env(args.env),
        policy,
        optimizer,
        discount=0.99,
        lambda_pi=args.lambda_pi,
        lambda_ve=args.lambda_ve,
        entropy_bonus=0.01,
        max_sample_length=10,
        actor_history_len=args.history_len,
        n_actors=16,
        max_timesteps=10000000,
        state_dtype=tf.uint8,
        log_every_n_steps=250000,
        grad_clip=40.,
    )


if __name__ == '__main__':
    main()
