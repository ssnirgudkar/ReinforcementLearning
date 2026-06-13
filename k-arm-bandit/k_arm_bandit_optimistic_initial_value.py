# Optimistic initial values for k-armed bandit problem
import numpy as np
import matplotlib.pyplot as plt


def generate_data_for_k_armed_bandit(num_arms, mean, variance):
    means = np.random.normal(loc=mean, scale=variance, size=num_arms)
    return (means)

def simulate_k_armed_bandit(num_arms, time_steps, means, average_reward_per_step, optimal_action_chosen_per_step, \
                            average_error_per_step, epsilon, index, config):
    Q = np.zeros(num_arms) # action-value estimates, we do not have any knowledge about the rewards, so we initialize them to zero
    # But here we arbitrarily set Q[0] to high value to encourage exploration of the first action, 
    # which is a common technique in optimistic initial values.
    if (epsilon == 0):
        Q[:] = 5
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
    epsilon_values = [0, 0.1]
    
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
    num_runs = 2000
    variance = 1
    simulate_multiple_runs_of_k_arm_bandit(num_runs, mean, variance, figName="figure_2_2_optimistic_initial_value.png")

if __name__ == "__main__":
    simulate_multiple_runs_of_k_arm_bandit_for_given_distribution()    

