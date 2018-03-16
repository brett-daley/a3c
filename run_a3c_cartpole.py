import gym
import tensorflow as tf
import a3c
import utils


def main():
    env = gym.make('CartPole-v0')
    # TODO: re-enable videos
    env = gym.wrappers.Monitor(env, 'videos/', force=True, video_callable=lambda e: False)

    utils.seed_all(env, seed=0)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

    a3c.execute(
        env=env,
        optimizer=optimizer,
        discount=0.99,
        max_sample_length=5,
        n_actors=16,
        n_iterations=75000,
    )


if __name__ == '__main__':
    main()
