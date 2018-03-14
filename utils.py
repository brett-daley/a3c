import tensorflow as tf
import numpy as np
import itertools


def seed_all(env, seed=None):
    if seed is None:
        seed = np.random.randint(65536)

    print('Using {} as random seed'.format(seed))

    env.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)


class Sampler:
    def __init__(self, env, discount, policy_func, value_func):
        self.env      = env
        self.state    = None
        self.done     = True
        self.discount = discount
        self.policy   = policy_func
        self.value    = value_func

    def sample(self, t_max):
        states, actions, rewards, done = self._sample(t_max)

        last_state_value = 0. if done else self.value(states[-1])
        returns = self._compute_returns(rewards, last_state_value)

        return states, actions, returns

    def _sample(self, t_max):
        states  = []
        actions = []
        rewards = []

        if self.done:
            self.state = self.env.reset()
            self.done = False

        for t in itertools.count():
            if t >= t_max or self.done:
                break

            states.append(self.state)

            action_distr = self.policy(self.state)
            action = np.random.choice(np.arange(self.env.action_space.n), p=action_distr)

            self.state, reward, self.done, _ = self.env.step(action)

            actions.append(action)
            rewards.append(reward)

        return np.array(states), np.array(actions), np.array(rewards), self.done

    def _compute_returns(self, rewards, last_state_value):
        for i in reversed(range(len(rewards) - 1)):
            rewards[i] += self.discount * rewards[i+1]
        return rewards
