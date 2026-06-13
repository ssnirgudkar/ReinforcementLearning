# k arm bandit upper confidence bound
# k arm bandit non-stationary problem

# For each time step, add some noise to the true action values (the means/q*)
import numpy as np
import matplotlib.pyplot as plt

def generate_data_for_k_armed_bandit(num_arms, mean, variance):
    means = np.random.normal(loc=mean, scale=variance, size=num_arms)
    return (means)


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

def simulate_k_arm_bandit(num_arms, time_steps, means, average_reward_per_step, optimal_action_chosen_per_step, \
                                         optimal_action_functor_index, config, optimal_action_functor):
    Q = np.zeros(num_arms) # action-value estimates, we do not have any knowledge about the rewards, so we initialize them to zero
    action_frequency = np.zeros(num_arms)

    if (optimal_action_functor == "ucb"):
        epsilon = 0
        func = calculate_optimal_action_using_ucb
    else:
        epsilon = 0.1
        func = calculate_optimal_action_using_epsilon_greedy

    for t in range(time_steps):
        if (optimal_action_functor == "ucb"):
            optimal_action = func(Q, action_frequency, t, c=2)
        else:
            optimal_action = func(Q, epsilon)

        # For each time step, add some noise to the true action values (the means/q*)
        reward = reward_from_environment(means, optimal_action)
        
        action_frequency[optimal_action] += 1

        # Update frequency then update estimate. This is important because the UCB calculation uses the frequency to calculate the confidence bound, so we need to update the frequency before updating the estimate.   
        Q[optimal_action] += ( reward - Q[optimal_action])/action_frequency[optimal_action]
        average_reward_per_step[config, t, optimal_action_functor_index] += reward
        optimal_action_chosen_per_step[config, t, optimal_action_functor_index] += (optimal_action == np.argmax(means))
        

    
def simulate_multiple_runs_of_k_arm_bandit(total_runs, mean, variance, figName):
    time_steps = 1000
    num_arms = 10
    optimal_action_functors = ["ucb", "epsilon_greedy"]

    average_reward_per_step = np.zeros((total_runs, time_steps, len(optimal_action_functors)))
    optimal_action_chosen_per_step = np.zeros((total_runs, time_steps, len(optimal_action_functors)))

    optimal_action_functors = ["ucb", "epsilon_greedy"]
    for config in range(total_runs):
        print("Run {0}/{1}".format(config+1, total_runs))
        means = generate_data_for_k_armed_bandit(num_arms, mean, variance)
        for index, optimal_action_functor in enumerate(optimal_action_functors):
            simulate_k_arm_bandit(num_arms, time_steps, means, average_reward_per_step, optimal_action_chosen_per_step, \
                                  index, config, optimal_action_functor)
       
            
    average_reward_per_step = average_reward_per_step.mean(axis=0)
    optimal_action_chosen_per_step = optimal_action_chosen_per_step.mean(axis=0) * 100

    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.set_xlabel("Number of steps", fontsize=12)
    ax2.set_xlabel("Number of steps", fontsize=12)
    ax1.set_ylabel("Average reward", fontsize=12)
    ax2.set_ylabel("% Optimal action chosen", fontsize=12)
    ax1.set_title(f"mean(μ)={mean}, variance(σ²)={variance}, number of runs={total_runs}", fontsize=12)
    
    for i, eps in enumerate(optimal_action_functors):
        ax1.plot(average_reward_per_step[:, i], label=f"ε = {eps}")
        ax2.plot(optimal_action_chosen_per_step[:, i], label=f"ε = {eps}")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(figName, dpi=150)
    plt.show()

if __name__ == "__main__":
    simulate_multiple_runs_of_k_arm_bandit(2000, 0, 1, "k_arm_bandit_ucb.png")

        