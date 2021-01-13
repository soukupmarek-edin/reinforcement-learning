import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bandits import Task

n_actions = 10
n_steps = 1000
n_runs = 1000

task_rewards = np.zeros((n_steps, n_runs))
optimal_actions = np.zeros((n_steps, n_runs))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plt.rc('figure', figsize=(12,5))

    for r in range(n_runs):
        task = Task(n_actions, n_steps, selection='upper_confidence_bound')
        task.run_task()
        task_rewards[:, r] = task.reward_tracker

    plt.plot(task_rewards.mean(axis=1))

    for r in range(n_runs):
        task = Task(n_actions, n_steps, selection='epsilon_greedy', epsilon=0.1)
        task.run_task()
        task_rewards[:, r] = task.reward_tracker

    plt.plot(task_rewards.mean(axis=1))
    plt.show()