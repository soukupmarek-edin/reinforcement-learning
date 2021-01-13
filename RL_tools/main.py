import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bandits import Task

n_actions = 10
n_steps = 1000
n_runs = 100

task_rewards = np.zeros((n_steps, n_runs))
optimal_actions = np.zeros((n_steps, n_runs))

if __name__ == '__main__':
    plt.rc('figure', figsize=(12, 5))

    for i in range(n_runs):
        task = Task('gradient', n_actions, n_steps,
                    rewards=np.random.normal(4, size=n_actions),
                    reward_estimates_init=np.zeros(n_actions),
                    step_size=0.1,
                    baseline='mean')
        task.run_task()
        optimal_actions[:, i] = task.optimal_action_tracker

    plt.plot(optimal_actions.mean(axis=1))

    for i in range(n_runs):
        task = Task('gradient', n_actions, n_steps,
                    rewards=np.random.normal(4, size=n_actions),
                    reward_estimates_init=np.zeros(n_actions),
                    step_size=0.4,
                    baseline='mean'
                    )
        task.run_task()
        optimal_actions[:, i] = task.optimal_action_tracker

    plt.plot(optimal_actions.mean(axis=1))

    plt.show()