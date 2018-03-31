import gym
import tensorflow as tf
import a3c
import utils


def main():
    env = gym.make('CartPole-v0')

    utils.seed_all(env, seed=0)

    objective_optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, decay=0.999)
    loss_optimizer      = tf.train.RMSPropOptimizer(learning_rate=1e-3, decay=0.999)

    a3c.execute(
        env,
        objective_optimizer,
        loss_optimizer,
        discount=0.99,
        entropy_bonus=0.01,
        max_sample_length=20,
        n_actors=16,
        max_timesteps=1000000,
    )


if __name__ == '__main__':
    main()
