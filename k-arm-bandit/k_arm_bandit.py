import numpy as np
import matplotlib.pyplot as plt


def generate_data_for_k_armed_bandit(num_arms, mean, variance):
    means = np.random.normal(loc=mean, scale=variance, size=num_arms)
    '''
    samples = np.array([np.random.normal(loc=m, scale=1, size=samples_per_action) for m in means])

    print("Shape of samples={0}".format(samples.shape)) # Should be (10,50)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (m, row) in enumerate(zip(means, samples)):
        action = i + 1
        jitter = np.random.uniform(-0.05, 0.05, size=samples_per_action)
        ax.scatter(np.full(samples_per_action, action) + jitter, row, s=18, alpha=0.5, color="steelblue", zorder=2)
        ax.hlines(m, action - 0.25, action + 0.25, colors="black", linewidths=2, zorder=3)
        ax.text(action, m, f" $q_*({action})$", va="center", ha="left", fontsize=8, color="black")

    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([str(i) for i in range(1, 11)])
    ax.set_xlabel("Actions", fontsize=12)
    ax.set_ylabel("Reward distribution", fontsize=12)
    ax.set_title("10-Armed Bandit: Reward Distributions per Action", fontsize=13)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("distributions.png", dpi=150)
    plt.show()
    print("Plot saved to distributions.png")
    '''
    return (means)

def simulate_k_armed_bandit(num_arms, time_steps, means, average_reward_per_step, optimal_action_chosen_per_step, \
                            average_error_per_step, epsilon, index, config):
    Q = np.zeros(num_arms) # action-value estimates, we do not have any knowledge about the rewards, so we initialize them to zero
    action_frequency = np.zeros(num_arms)
    
    for t in range(time_steps):
        # based on current estimates, select the action with the highest reward
        # With probability epsilon, select random action (exploration). 
        # With probability 1-epsilon, select action with highest estimated reward (exploitation)
        if np.random.rand() < epsilon:
            optimal_action = np.random.choice(num_arms)
        else:
            optimal_action = np.argmax(Q) # we cannot use samples here because it will be like looking at all arms simultaneously, 
        # which is not how the bandit problem works. We need to select one action based on our current estimates Q, 
        # and then observe the reward for that action.
        # Take the normal distribution corresponding to the optimal action and sample a reward from it
        # One can generate samples data and use it here, but using np.random.normal is more realistic 
        # because it simulates the process of taking an action and observing a reward, which is what happens in the bandit problem. 
        reward = np.random.normal(means[optimal_action], 1) #samples[optimal_action, t]
        average_reward_per_step[config, t, index] += reward
        optimal_action_chosen_per_step[config, t, index] += (optimal_action == np.argmax(means)) # we check if the action we selected is the optimal action (the one with the highest mean reward)
        #update the estimates for the optimal action
        action_frequency[optimal_action] += 1

        average_error_per_step[config, t, index] += (reward - Q[optimal_action])
        
        # Incremental update of the action-value estimate for the optimal action using the observed reward
        Q[optimal_action] += (reward - Q[optimal_action])/action_frequency[optimal_action]
        
        

    #print("Estimated action values Q={0}".format(Q))
    #print("Action frequencies={0}".format(action_frequency))

def simulate_multiple_runs_of_k_arm_bandit(total_runs, mean, variance, figName):
    time_steps = 1000
    num_arms=10
    epsilon_values = [0, 0.01, 0.1]
    
    # Allocate
    average_reward_per_step = np.zeros((total_runs, time_steps, len(epsilon_values)))
    optimal_action_chosen_per_step = np.zeros((total_runs, time_steps, len(epsilon_values)))
    average_error_per_step = np.zeros((total_runs, time_steps, len(epsilon_values)))
    for config in range(total_runs):
        print("Run {0}/{1}".format(config+1, total_runs))
        means = generate_data_for_k_armed_bandit(num_arms, mean, variance)
        for index, epsilon in enumerate(epsilon_values):
            simulate_k_armed_bandit(num_arms, time_steps, means, average_reward_per_step, optimal_action_chosen_per_step, \
                                    average_error_per_step, epsilon, index, config)
    average_reward_per_step = average_reward_per_step.mean(axis=0)
    optimal_action_chosen_per_step = optimal_action_chosen_per_step.mean(axis=0) * 100 # convert to percentage
    average_error_per_step = average_error_per_step.mean(axis=0)
    # Plotting the results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    ax1.set_xlabel("Number of steps", fontsize=12)
    ax2.set_xlabel("Number of steps", fontsize=12)
    ax3.set_xlabel("Number of steps", fontsize=12)
    ax1.set_ylabel("Average reward", fontsize=12)
    ax2.set_ylabel("% Optimal action chosen", fontsize=12)
    ax3.set_ylabel("Average error per time step", fontsize=12)
    ax1.set_title(f"mean(μ)={mean}, variance(σ²)={variance}, number of runs={total_runs}", fontsize=12)
    
    for i, eps in enumerate(epsilon_values):
        ax1.plot(average_reward_per_step[:, i], label=f"ε = {eps}")
        ax2.plot(optimal_action_chosen_per_step[:, i], label=f"ε = {eps}")
        ax3.plot(average_error_per_step[:, i], label=f"ε = {eps}")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.tight_layout()
    plt.savefig(figName, dpi=150)
    plt.show()

def simulate_multiple_runs_of_k_arm_bandit_for_given_distribution():   
    mean = 0
    variance = 0.1 
    num_runs = 1000
    # low variance means that the rewards are more consistent and less noisy, which allows 
    # the greedy method (ε=0) to quickly identify and exploit the optimal action. 
    # In contrast, with higher variance, the rewards are more spread out and noisy, making it harder for the greedy method to identify the optimal action, while the epsilon-greedy method can still explore and find it over time.
    simulate_multiple_runs_of_k_arm_bandit(num_runs, mean, variance, figName="figure_2_2_low_variance.png")
    variance = 1 
    simulate_multiple_runs_of_k_arm_bandit(num_runs, mean, variance, figName="figure_2_2_medium_variance.png")
    variance = 10
    simulate_multiple_runs_of_k_arm_bandit(num_runs, mean, variance, figName="figure_2_2_high_variance.png")

if __name__ == "__main__":
    simulate_multiple_runs_of_k_arm_bandit_for_given_distribution()    

# Experiment with low variance and see if greedy method wins over epsilon greedy
# Experiment with non-stationary distributions and see if epsilon greedy wins over greedy method    