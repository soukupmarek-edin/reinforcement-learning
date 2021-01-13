import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bandits import Task

n_actions = 10
n_steps = 1000
n_runs = 1000

epsilons = [0, 0.01, 0.1]
eps_rewards = np.zeros((n_steps, len(epsilons)))
eps_optimal_actions = np.zeros((n_steps, len(epsilons)))

task_rewards = np.zeros((n_steps, n_runs))
optimal_actions = np.zeros((n_steps, n_runs))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for i, epsilon in enumerate(epsilons):
        for j in range(n_runs):
            task = Task(n_actions, n_steps, epsilon=epsilon)
            task.run_task()
            task_rewards[:, j] = task.reward_tracker
            optimal_actions[:, j] = task.optimal_action_tracker

        eps_rewards[:, i] = task_rewards.mean(axis=1)
        eps_optimal_actions[:, i] = optimal_actions.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    rewards_df = pd.DataFrame(eps_rewards, columns=epsilons)
    rewards_df.plot(ax=ax1)

    opt_actions_df = pd.DataFrame(eps_optimal_actions, columns=epsilons)
    opt_actions_df.plot(ax=ax2)
    ax2.set_ylim(ymax=1)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.5)
    plt.show()