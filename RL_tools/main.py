import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bandits import Task

n_actions = 10
n_steps = 1000
n_runs = 500

task_rewards = np.zeros((n_steps, n_runs))
optimal_actions = np.zeros((n_steps, n_runs))

if __name__ == '__main__':
    plt.rc('figure', figsize=(12, 5))

    for step_size in [0.1, 0.4]:
        for i in range(n_runs):
            task = Task('gradient', n_actions, n_steps,
                        rewards=np.random.normal(0, size=n_actions),
                        reward_estimates_init=np.zeros(n_actions),
                        step_size=step_size,
                        use_baseline=True)
            task.run_task()
            optimal_actions[:, i] = task.optimal_action_tracker

        plt.plot(optimal_actions.mean(axis=1))

    plt.show()