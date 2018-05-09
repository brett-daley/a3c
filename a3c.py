import tensorflow as tf
import numpy as np
import itertools
import gym
import threading
import math
import time
import utils
import wrappers


def execute(
        env,
        policy,
        optimizer,
        discount,
        Lambda,
        entropy_bonus,
        max_sample_length,
        actor_history_len,
        n_actors,
        max_timesteps,
        wrapper=None,
        state_dtype=tf.float32,
        log_every_n_steps=25000,
    ):

    def prepare_env(e, video=False):
        e = gym.wrappers.Monitor(e, 'videos/', force=True)
        if not video:
            e.video_callable = lambda episode: False
        if wrapper is not None:
            e = wrapper(e)
        e = wrappers.HistoryWrapper(e, actor_history_len)
        e.seed(utils.random_seed())
        return e

    training_envs = [prepare_env(gym.make(env.spec.id)) for i in range(n_actors)]
    env = prepare_env(env, video=True)

    input_shape = list(env.observation_space.shape)
    input_shape[-1] *= actor_history_len
    n_actions   = env.action_space.n

    with tf.Session() as session:
        state_ph      = tf.placeholder(state_dtype, [None] + input_shape)
        action_ph     = tf.placeholder(tf.int32,    [None])
        return_ph     = tf.placeholder(tf.float32,  [None])

        action_distr, value = policy(state_ph, n_actions, scope='policy')
        policy_vars = tf.trainable_variables(scope='policy')

        action_indices = tf.stack([tf.range(tf.size(action_ph)), action_ph], axis=1)
        action_probs = tf.gather_nd(action_distr, action_indices)
        log_action_probs = tf.log(action_probs + 1e-10)

        objective = tf.reduce_sum(log_action_probs * (return_ph - tf.stop_gradient(value)))
        loss      = 0.5 * tf.reduce_sum(tf.square(return_ph - value))
        entropy   = entropy_bonus * (-tf.reduce_sum(log_action_probs * action_probs))

        total_loss = -(objective + entropy) + loss

        grads = tf.gradients(total_loss, policy_vars)
        grads, _ = tf.clip_by_global_norm(grads, 100.)
        grads_and_vars = list(zip(grads, policy_vars))
        train_op = optimizer.apply_gradients(grads_and_vars)

        session.run(tf.global_variables_initializer())

        shared_counter = utils.Counter(period=log_every_n_steps)


        class Actor:
            def __init__(self, env, counter):
                self.env = env

                self.state   = None
                self.done    = True

                self.counter = counter
                self.thread  = None

            def start(self):
                self.thread = threading.Thread(target=self._train)
                self.thread.start()

            def join(self):
                assert self.thread is not None
                self.thread.join()

            def _train(self):
                while not self.counter.is_expired():
                    states, actions, returns = self._sample()

                    self.counter.increment(len(states))

                    for i in range(0, len(states), max_sample_length):
                        session.run(train_op, feed_dict={
                            state_ph:  states[i:i+max_sample_length],
                            action_ph: actions[i:i+max_sample_length],
                            return_ph: returns[i:i+max_sample_length],
                        })

            def policy(self, state):
                distr = session.run(action_distr, feed_dict={state_ph: state[None]})[0]
                action = np.random.choice(np.arange(n_actions), p=distr)
                return action

            def _value(self, states):
                return session.run(value, feed_dict={state_ph: states})

            def _sample(self):
                states  = []
                actions = []
                rewards = []

                state = self.state
                done = self.done

                if done:
                    state = self.env.reset()
                    done = False

                states.append(state)

                for t in itertools.count():
                    if done:
                        break

                    action = self.policy(state)

                    state, reward, done, _ = self.env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                self.state = state
                self.done = done

                states  = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)

                return states[:-1], actions, self._compute_returns(states, rewards)

            def _compute_returns(self, states, rewards):
                values = self._value(states)
                if self.done:
                    values[-1] = 0.
                returns = rewards + (discount * values[1:])

                for i in reversed(range(len(returns) - 1)):
                    returns[i] += (discount * Lambda) * (returns[i+1] - values[i+1])

                return returns

            def get_n_episodes(self):
                return len(utils.get_episode_rewards(self.env))


        def benchmark(actor, n_episodes):
            for i in range(n_episodes):
                state = env.reset()
                done = False

                while not done:
                    action = actor.policy(state)
                    state, _, done, _ = env.step(action)

            rewards = utils.get_episode_rewards(env)[-n_episodes:]

            return np.mean(rewards), np.std(rewards)


        actors = [Actor(training_envs[i], shared_counter) for i in range(n_actors)]
        timesteps = 0
        best_mean_reward = -float('inf')
        start_time = time.time()

        for e in itertools.count():
            print('Epoch', e)
            print('Timestep', timesteps)
            print('Realtime {:.3f}'.format(time.time() - start_time))
            print('Episodes', sum([a.get_n_episodes() for a in actors]))

            mean_reward, std_reward = benchmark(actors[0], n_episodes=30)
            best_mean_reward = max(mean_reward, best_mean_reward)

            print('Mean reward', mean_reward)
            print('Best mean reward', best_mean_reward)
            print('Standard dev', std_reward)
            print(flush=True)

            if timesteps >= max_timesteps:
                break

            for a in actors:
                a.start()

            for a in actors:
                a.join()

            timesteps += shared_counter.value()
            shared_counter.reset()
