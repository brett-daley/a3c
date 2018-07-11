import gym
import tensorflow as tf
import argparse

import a3c
import utils
import policies
import wrappers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',         type=str,   default='PongNoFrameskip-v3')
    parser.add_argument('--Lambda',      type=float, default=1.0)
    parser.add_argument('--history_len', type=int,   default=4)
    args = parser.parse_args()

    env = gym.make(args.env)

    utils.seed_all(seed=0)

    policy = policies.AtariPolicy

    optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, use_locking=True)

    a3c.execute(
        env,
        policy,
        optimizer,
        discount=0.99,
        Lambda=args.Lambda,
        entropy_bonus=0.01,
        max_sample_length=10,
        actor_history_len=args.history_len,
        n_actors=16,
        wrapper=wrappers.wrap_deepmind,
        max_timesteps=10000000,
        state_dtype=tf.uint8,
        log_every_n_steps=250000,
        grad_clip=40.,
    )


if __name__ == '__main__':
    main()
