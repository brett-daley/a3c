import gym
import tensorflow as tf
import a3c
import utils
import policies
import wrappers


def main():
    env = gym.make(gym.benchmark_spec('Atari200M').tasks[3].env_id)

    utils.seed_all(seed=0)

    policy = policies.AtariPolicy

    optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.99,
        use_locking=True,
        epsilon=1e-4,
    )

    a3c.execute(
        env,
        policy,
        optimizer,
        discount=0.99,
        Lambda=1.0,
        entropy_bonus=0.01,
        max_sample_length=5,
        actor_history_len=4,
        n_actors=16,
        wrapper=wrappers.wrap_deepmind,
        max_timesteps=5000000,
        state_dtype=tf.uint8,
    )


if __name__ == '__main__':
    main()
