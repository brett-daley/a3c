import gym
import tensorflow as tf
import a3c
import utils


def main():
    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env, 'videos/', force=True)

    utils.seed_all(env, seed=0)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    a3c.execute(
        env=env,
        optimizer=optimizer,
        discount=0.99,
        max_sample_length=100,
        n_iterations=100000,
    )


if __name__ == '__main__':
    main()
