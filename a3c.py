import tensorflow as tf
import numpy as np
import itertools
import gym
import threading
import math
import time
import utils


def execute(
        env,
        policy,
        objective_optimizer,
        loss_optimizer,
        discount,
        entropy_bonus,
        max_sample_length,
        n_actors,
        max_timesteps,
        wrapper=None,
        state_dtype=tf.float32,
        log_every_n_steps=25000,
    ):

    def prepare_env(e):
        if wrapper is not None:
            e = wrapper(e)
        e = gym.wrappers.Monitor(e, 'videos/', force=True, video_callable=lambda episode: False)
        e.seed(utils.random_seed())
        return e

    training_envs = [prepare_env(gym.make(env.spec.id)) for i in range(n_actors)]
    env = prepare_env(env)

    input_shape = env.observation_space.shape
    n_actions   = env.action_space.n

    with tf.Session() as session:
        state_ph      = tf.placeholder(state_dtype, [None] + list(input_shape))
        action_ph     = tf.placeholder(tf.int32,    [None])
        return_ph     = tf.placeholder(tf.float32,  [None])

        action_distr, value = policy(state_ph, n_actions, scope='policy')
        policy_vars = tf.trainable_variables(scope='policy')

        action_indices = tf.stack([tf.range(tf.size(action_ph)), action_ph], axis=1)
        action_probs = tf.gather_nd(action_distr, action_indices)
        log_action_probs = tf.log(action_probs + 1e-30) / tf.log(2.)

        objective = tf.reduce_sum(log_action_probs * (return_ph - tf.stop_gradient(value)))
        loss      = tf.reduce_sum(tf.square(return_ph - value))
        entropy   = entropy_bonus * (-tf.reduce_sum(log_action_probs * action_probs))

        grads_and_vars     = objective_optimizer.compute_gradients(-(objective + entropy), var_list=policy_vars)
        objective_train_op = objective_optimizer.apply_gradients(grads_and_vars)

        grads_and_vars = loss_optimizer.compute_gradients(loss, var_list=policy_vars)
        loss_train_op  = loss_optimizer.apply_gradients(grads_and_vars)

        train_op = tf.group(*[objective_train_op, loss_train_op])

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

                    session.run(train_op, feed_dict={
                        state_ph:  states,
                        action_ph: actions,
                        return_ph: returns,
                    })

            def policy(self, state):
                distr = session.run(action_distr, feed_dict={state_ph: state[None]})[0]
                action = np.random.choice(np.arange(n_actions), p=distr)
                return action

            def _value(self, state):
                return session.run(value, feed_dict={state_ph: state[None]})[0]

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
                    if t == max_sample_length or done:
                        break

                    action = self.policy(state)

                    state, reward, done, _ = self.env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                self.state = state
                self.done = done

                return np.array(states[:-1]), np.array(actions), self._compute_returns(rewards)

            def _compute_returns(self, rewards):
                last_value = 0. if self.done else self._value(self.state)
                values = last_value * np.array([discount**(i+1) for i in reversed(range(len(rewards)))])

                for i in reversed(range(len(rewards) - 1)):
                    rewards[i] += discount * rewards[i+1]

                return (rewards + values)

            def get_n_episodes(self):
                return len(self.env.get_episode_rewards())


        def benchmark(actor, n_episodes):
            for i in range(n_episodes):
                state = env.reset()
                done = False

                while not done:
                    action = actor.policy(state)
                    state, _, done, _ = env.step(action)

            rewards = env.get_episode_rewards()[-n_episodes:]

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
