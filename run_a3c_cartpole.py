import gym
import tensorflow as tf
import a3c
import utils


def main():
    env = gym.make('CartPole-v0')

    utils.seed_all(env, seed=0)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

    a3c.execute(
        env=env,
        optimizer=optimizer,
        discount=0.99,
        entropy_penalty=0.01,
        max_sample_length=5,
        n_actors=16,
        max_timesteps=300000,
    )


if __name__ == '__main__':
    main()
