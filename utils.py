import tensorflow as tf
import numpy as np
import itertools
import copy


def seed_all(env, seed=None):
    if seed is None:
        seed = np.random.randint(65536)

    print('Using {} as random seed'.format(seed))

    env.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)


class Actor:
    def __init__(self, env, discount, policy_func, value_func):
        self.env      = copy.deepcopy(env)
        self.state    = None
        self.done     = True
        self.discount = discount
        self.policy   = policy_func
        self.value    = value_func

    def sample(self, t_max):
        states, actions, rewards, last_value = self._sample(t_max)
        returns = self._compute_returns(rewards, last_value)

        return states, actions, returns

    def _sample(self, t_max):
        states  = []
        actions = []
        rewards = []

        if self.done:
            self.state = self.env.reset()
            self.done = False

        states.append(self.state)

        for t in itertools.count():
            action = self.policy(self.state)

            self.state, reward, self.done, _ = self.env.step(action)

            states.append(self.state)
            actions.append(action)
            rewards.append(reward)

            if t == t_max or self.done:
                break

        last_value = 0. if self.done else self.value(states[-1])

        return np.array(states[:-1]), np.array(actions), np.array(rewards), last_value

    def _compute_returns(self, rewards, last_value):
        values = last_value * np.array([self.discount**i for i in range(1, len(rewards)+1)])

        for i in reversed(range(len(rewards) - 1)):
            rewards[i] += self.discount * rewards[i+1]

        return rewards + values

    def get_n_episodes(self):
        return len(self.env.get_episode_rewards())

    def get_average_reward(self, n):
        return np.mean(self.env.get_episode_rewards()[-n:])
