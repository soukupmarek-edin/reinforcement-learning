import numpy as np


class Bandit:

    def __init__(self, n_actions, n_steps, reward_estimates_init, step_size='mean', epsilon=0.01):
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.step_size = step_size
        self.epsilon = epsilon

        self.n_action_selected = np.zeros(n_actions)
        self.reward_estimates = reward_estimates_init
        self.A = 0

    def select_action(self):
        rnd = np.random.uniform()
        if self.epsilon > rnd:
            A = np.random.choice(np.arange(self.n_actions))
        else:
            A = self.reward_estimates.argmax()

        self.n_action_selected[A] += 1
        self.A = A
        return A

    def update_reward_estimate(self, reward):
        Q_a = self.reward_estimates[self.A]

        if self.step_size == 'mean':
            N_a = self.n_action_selected[self.A]
            alpha = 1 / N_a
        else:
            alpha = self.step_size

        self.reward_estimates[self.A] = Q_a + alpha * (reward - Q_a)


class Task:

    def __init__(self, n_actions, n_steps, **kwargs):
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.rewards = np.random.normal(size=n_actions)

        if 'step_size' in kwargs.keys():
            step_size = kwargs['step_size']
        else:
            step_size = 'mean'

        if 'epsilon' in kwargs.keys():
            epsilon = kwargs['epsilon']
        else:
            epsilon = 0.

        if 'reward_estimates_init' not in kwargs.keys():
            reward_estimates_init = np.zeros(n_actions)
        else:
            reward_estimates_init = kwargs['reward_estimates_init']

        self.optimal_action = self.rewards.argmax()
        self.step = 0

        self.bandit = Bandit(n_actions, n_steps, reward_estimates_init, step_size, epsilon)

        self.reward_tracker = np.zeros(n_steps)
        self.optimal_action_tracker = np.zeros(n_steps)

    def make_step(self):
        A = self.bandit.select_action()
        reward = self.rewards[A] + np.random.normal()
        self.reward_tracker[self.step] = reward

        if A == self.optimal_action:
            self.optimal_action_tracker[self.step] += 1
        self.bandit.update_reward_estimate(reward)

    def run_task(self):
        for _ in range(self.n_steps):
            self.make_step()
            self.step += 1