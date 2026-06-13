# k arm bandit non-stationary problem

# For each time step, add some noise to the true action values (the means/q*)
import numpy as np
import matplotlib.pyplot as plt

def generate_data_for_k_armed_bandit(num_arms, mean, variance):
    means = np.random.normal(loc=mean, scale=variance, size=num_arms)
    return (means)

def add_noise_to_means(means, noise_mean, noise_variance):
    noise = np.random.normal(loc=noise_mean, scale=noise_variance, size=means.shape)
    return means + noise

# How the environment calculates reward is opaque to the agent
def reward_from_environment_nonstationary(means, optimal_action):
    return np.random.normal(add_noise_to_means(means[optimal_action], 0, 0.01), 1)

# For stationary k arm bandit
def reward_from_environment(means, optimal_action):
    return np.random.normal(means[optimal_action], 1)

# Using upper confidence bound (UCB)
# At = argmax_a [ Q(a) + c * sqrt(ln(t)/N(a)) ]
def calculate_optimal_action_using_ucb(Q, action_frequency, t, c):
    ucb_values = Q + c * np.sqrt(np.log(t + 1) / (action_frequency + 1e-5)) # add small value to avoid division by zero
    return np.argmax(ucb_values)

def calculate_optimal_action_using_epsilon_greedy(Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q))
    else:
        return np.argmax(Q)

def simulate_k_arm_bandit_nonstationary(num_arms, time_steps, means, average_reward_per_step, optimal_action_chosen_per_step, \
                                        epsilon, alpha, epsilon_index, config):
    Q = np.zeros(num_arms) # action-value estimates, we do not have any knowledge about the rewards, so we initialize them to zero
    action_frequency = np.zeros(num_arms)

    for t in range(time_steps):
        optimal_action = calculate_optimal_action_using_epsilon_greedy(Q, epsilon)

        # For each time step, add some noise to the true action values (the means/q*)
        reward = reward_from_environment(means, optimal_action)
        
        action_frequency[optimal_action] += 1

        Q[optimal_action] += alpha * ( reward - Q[optimal_action])
        average_reward_per_step[config, t, epsilon_index] += reward
        optimal_action_chosen_per_step[config, t, epsilon_index] += (optimal_action == np.argmax(means))
        

def simulate_multiple_runs_of_k_arm_bandit(total_runs, mean, variance, figName):
    time_steps = 10000
    num_arms = 10
    epsilon_values = [0, 0.1]
    alpha = 0.1 # step size parameter for non-stationary problem, we use a constant step size to give more weight to recent rewards, which is important in a non-stationary problem where the true action values can change over time.

    average_reward_per_step = np.zeros((total_runs, time_steps, len(epsilon_values)))
    optimal_action_chosen_per_step = np.zeros((total_runs, time_steps, len(epsilon_values)))

    for config in range(total_runs):
        print("Run {0}/{1}".format(config+1, total_runs))
        means = generate_data_for_k_armed_bandit(num_arms, mean, variance)
        for index, epsilon in enumerate(epsilon_values):
            simulate_k_arm_bandit_nonstationary(num_arms, time_steps, means, average_reward_per_step, optimal_action_chosen_per_step, \
                                                epsilon, alpha, index, config)
            
    average_reward_per_step = average_reward_per_step.mean(axis=0)
    optimal_action_chosen_per_step = optimal_action_chosen_per_step.mean(axis=0) * 100

    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.set_xlabel("Number of steps", fontsize=12)
    ax2.set_xlabel("Number of steps", fontsize=12)
    ax1.set_ylabel("Average reward", fontsize=12)
    ax2.set_ylabel("% Optimal action chosen", fontsize=12)
    ax1.set_title(f"mean(μ)={mean}, variance(σ²)={variance}, number of runs={total_runs}", fontsize=12)
    
    for i, eps in enumerate(epsilon_values):
        ax1.plot(average_reward_per_step[:, i], label=f"ε = {eps}")
        ax2.plot(optimal_action_chosen_per_step[:, i], label=f"ε = {eps}")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(figName, dpi=150)
    plt.show()

if __name__ == "__main__":
    simulate_multiple_runs_of_k_arm_bandit(2000, 0, 1, "k_arm_bandit_nonstationary.png")

        