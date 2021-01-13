import numpy as np


class Bandit:

    def __init__(self, n_actions, n_steps, reward_estimates_init, selection, step_size, epsilon, c):
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.step_size = step_size
        self.selection = selection
        self.epsilon = epsilon
        self.c = c

        self.n_action_selected = np.ones(n_actions)/1e5
        self.reward_estimates = reward_estimates_init
        self.action = 0
        self.step = 1

    def epsilon_greedy(self):
        rnd = np.random.uniform()
        if self.epsilon > rnd:
            action = np.random.choice(np.arange(self.n_actions))
        else:
            action = self.reward_estimates.argmax()

        self.n_action_selected[action] += 1
        self.action = action
        return action

    def upper_confidence_bound(self):

        action = np.argmax(self.reward_estimates + self.c * np.sqrt(np.log(self.step) / self.n_action_selected))

        self.n_action_selected[action] += 1
        self.action = action
        return action

    def select_action(self):
        if self.selection == 'epsilon_greedy':
            return self.epsilon_greedy()
        elif self.selection == 'upper_confidence_bound':
            return self.upper_confidence_bound()
        else:
            raise AttributeError("Unknown selection function")

    def update_reward_estimate(self, new_reward):
        current_reward = self.reward_estimates[self.action]

        if self.step_size == 'mean':
            n_action = self.n_action_selected[self.action]
            alpha = 1 / n_action
        else:
            alpha = self.step_size

        self.reward_estimates[self.action] = current_reward + alpha * (new_reward - current_reward)
        self.step += 1


class Task:

    def __init__(self, n_actions, n_steps, **kwargs):
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.rewards = np.random.normal(size=n_actions)

        if 'reward_estimates_init' in kwargs.keys():
            reward_estimates_init = kwargs['reward_estimates_init']
        else:
            reward_estimates_init = np.zeros(n_actions)

        if 'selection' in kwargs.keys():
            selection = kwargs['selection']
        else:
            selection = 'epsilon_greedy'

        if 'step_size' in kwargs.keys():
            step_size = kwargs['step_size']
        else:
            step_size = 'mean'

        if 'epsilon' in kwargs.keys():
            epsilon = kwargs['epsilon']
        else:
            epsilon = 0.

        if 'c' in kwargs.keys():
            c = kwargs['c']
        else:
            c = 2

        self.optimal_action = self.rewards.argmax()
        self.step = 1

        self.bandit = Bandit(n_actions, n_steps, reward_estimates_init, selection, step_size, epsilon, c)

        self.reward_tracker = np.zeros(n_steps)
        self.optimal_action_tracker = np.zeros(n_steps)

    def make_step(self):
        action = self.bandit.select_action()
        reward = self.rewards[action] + np.random.normal()
        self.reward_tracker[self.step-1] = reward

        if action == self.optimal_action:
            self.optimal_action_tracker[self.step-1] += 1
        self.bandit.update_reward_estimate(reward)

    def run_task(self):
        for _ in range(self.n_steps):
            self.make_step()
            self.step += 1
