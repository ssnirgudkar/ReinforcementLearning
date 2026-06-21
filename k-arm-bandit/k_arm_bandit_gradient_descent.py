# Gradient bandit algorithm
import numpy as np
import matplotlib.pyplot as plt

def calculate_action_probabilities(H):
    # pi(a) = exp(H(a)) / sum_b exp(H(b))
    exp_H = np.exp(H ) # subtract max for numerical stability
    return exp_H / np.sum(exp_H)

def update_preference(H, action_taken, R, baseline, pi, alpha):
    # H_t+1(A) = H_t(A) + alpha * (R_t - baseline) * (1(a) - pi_t(a))
    # H_t+1(a) = H_t(a) - alpha * (R_t - baseline) * pi_t(a) for all a != A
    pi = calculate_action_probabilities(H)
    for a in range(len(H)):
        
        if a == action_taken:
            H[a] += alpha * (R - baseline) * (1 - pi[a])
        else:
            H[a] -= alpha * (R - baseline) * pi[a]

    return H

def update_baseline_reward(baseline, R, alpha):
    # baseline_t+1 = baseline_t + alpha * (R_t - baseline_t)
    return baseline + alpha * (R - baseline)

def generate_data_for_k_armed_bandit(num_arms, mean, variance):
    means = np.random.normal(loc=mean, scale=variance, size=num_arms)
    return (means)

# For stationary k arm bandit
def reward_from_environment(means, optimal_action):
    return np.random.normal(means[optimal_action], 1)

def simulate_multiple_runs_of_k_arm_bandit(num_of_runs, alpha, mean, variance, figName):
    time_steps = 1000
    num_arms = 10
    # last 2 dimensions are for : with baseline and without baseline
    average_reward_per_step = np.zeros((num_of_runs, time_steps, 2))
    optimal_action_chosen_per_step = np.zeros((num_of_runs, time_steps, 2))
    A = np.zeros(num_arms) # action estimates
    
    
    for config in range(2):
        baseline = 0 # baseline reward, we initialize it to zero, but it can be initialized to any value. 
        # The baseline is used to reduce the variance of the gradient estimates, which can help the algorithm converge faster. 
        for run in range(num_of_runs):
            print("Run {0}/{1}".format(run+1, num_of_runs))
            H = np.zeros(num_arms) # action preferences
            means = generate_data_for_k_armed_bandit(num_arms, mean, variance)
            for t in range(time_steps):
                if t == 0:
                    action_taken = np.random.choice(num_arms)
                else:
                    action_taken = np.random.choice(num_arms, p=calculate_action_probabilities(H))

                reward = reward_from_environment(means, action_taken)
                optimal_action_chosen_per_step[run, t, config] += (action_taken == np.argmax(means))
                average_reward_per_step[run, t, config] += reward

                
                H = update_preference(H, action_taken, reward, baseline, calculate_action_probabilities(H), alpha)
                if (config == 0): # with baseline
                    baseline = update_baseline_reward(baseline, reward, alpha)

    print("Shape of average_reward_per_step: ", average_reward_per_step.shape)
    print("Shape of optimal_action_chosen_per_step: ", optimal_action_chosen_per_step.shape)
    average_reward_per_step = np.mean(average_reward_per_step, axis=0)
    optimal_action_chosen_per_step = np.mean(optimal_action_chosen_per_step, axis=0) * 100
    print("Shape of average_reward_per_step after mean: ", average_reward_per_step.shape)
    print("Shape of optimal_action_chosen_per_step after mean: ", optimal_action_chosen_per_step.shape)
    

    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.set_xlabel("Number of steps", fontsize=12)
    ax2.set_xlabel("Number of steps", fontsize=12)
    ax1.set_ylabel("Average reward", fontsize=12)
    ax2.set_ylabel("% Optimal action chosen", fontsize=12)
    ax1.set_title(f"mean(μ)={mean}, variance(σ²)={variance}, number of runs={num_of_runs}", fontsize=12)
    
    for i in range(2):
        ax1.plot(average_reward_per_step[:,i], label=f"alpha = {alpha}")
        ax2.plot(optimal_action_chosen_per_step[:,i], label=f"alpha = {alpha}")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(figName, dpi=150)
    plt.show()


if __name__ == "__main__":
    alphas = [0.1, 0.4]
    mean = 4
    variance = 1
    for alpha in alphas:
        simulate_multiple_runs_of_k_arm_bandit(2000, alpha, mean, variance, "k_arm_bandit_gradient_descent_optimal_action" + "_"
                                               + str(alpha) + ".png")