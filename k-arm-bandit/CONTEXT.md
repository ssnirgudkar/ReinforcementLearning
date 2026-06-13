# K-Arm Bandit — Session Context

## What was implemented

Epsilon-greedy k-arm bandit from Sutton & Barto "Reinforcement Learning: An Introduction", Chapter 2.

File: `k_arm_bandit.py`

### Key design decisions

- `generate_data_for_k_armed_bandit(num_arms)` — generates arm means from N(0,1). Distribution plotting code is present but commented out (was used earlier to produce `distributions.png`).
- `simulate_k_armed_bandit(...)` — runs one episode: epsilon-greedy action selection, incremental Q update using sample averages, tracks reward and whether optimal action was chosen.
- `simulate_multiple_runs_of_k_arm_bandit(total_runs=2000)` — outer loop over 2000 runs × 3 epsilon values. Arm means are generated **once per run** and shared across all epsilon variants for a fair comparison. Results stored in 3D arrays `(runs, time_steps, epsilons)`, averaged over axis=0, then plotted.

### Output

Produces `figure_2_2.png` — two-panel plot matching Figure 2.2 from the book:
- Top: average reward over 1000 steps for ε = 0, 0.01, 0.1
- Bottom: % optimal action chosen over 1000 steps for ε = 0, 0.01, 0.1

