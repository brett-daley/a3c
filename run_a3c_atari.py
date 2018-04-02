import gym
import tensorflow as tf
import a3c
import utils
import policies


def main():
    env = gym.make(gym.benchmark_spec('Atari40M').tasks[3].env_id)
    import wrappers
    env = wrappers.wrap_deepmind(env)

    utils.seed_all(seed=0)

    policy = policies.AtariPolicy

    objective_optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5, decay=0.999)
    loss_optimizer      = tf.train.RMSPropOptimizer(learning_rate=1e-4, decay=0.99)

    a3c.execute(
        env,
        policy,
        objective_optimizer,
        loss_optimizer,
        discount=0.99,
        entropy_bonus=0.01,
        max_sample_length=20,
        n_actors=16,
        max_timesteps=40000000,
        state_dtype=tf.uint8,
    )


if __name__ == '__main__':
    main()
