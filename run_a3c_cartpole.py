import gym
import tensorflow as tf
import a3c
import utils
import policies


def main():
    env_name = 'CartPole-v0'

    utils.seed_all(seed=0)

    policy = policies.CartPolePolicy

    optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, use_locking=True)

    a3c.execute(
        lambda: gym.make(env_name),
        policy,
        optimizer,
        discount=0.99,
        entropy_bonus=0.01,
        max_sample_length=20,
        actor_history_len=1,
        n_actors=16,
        max_timesteps=1000000,
    )


if __name__ == '__main__':
    main()
