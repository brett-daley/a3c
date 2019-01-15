import gym
import tensorflow as tf
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
    env_name = 'pong'

    utils.seed_all(seed=0)

    policy = policies.AtariPolicy

    optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, use_locking=True)

    a3c.execute(
        lambda: make_atari_env(env_name),
        policy,
        optimizer,
        discount=0.99,
        entropy_bonus=0.01,
        max_sample_length=10,
        actor_history_len=4,
        n_actors=16,
        max_timesteps=10000000,
        state_dtype=tf.uint8,
        log_every_n_steps=250000,
        grad_clip=40.,
    )


if __name__ == '__main__':
    main()
