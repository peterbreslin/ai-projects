import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Create the Frozen Lake environment
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="ansi")

# Initialize the Q-table
qtable = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
N = 1000
alpha = 0.5
gamma = 0.9
n_success = 0

# List of outcomes to plot
outcomes = []

print("Q-table before training:\n", qtable)

# Training
for i in range(N):
    state, info = env.reset()
    terminated = False
    truncated = False

    # By default, we consider our outcomes to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not terminated or truncated:
        # Choose the action with the highest value in the current state
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = env.action_space.sample()

        #  Implement this action and move the agent in the desired direction
        new_state, reward, terminated, truncated, info = env.step(action)

        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        # Update current state
        state = new_state

        # If the agent reaches the goal, mark the outcome as a success
        if terminated and reward == 1.0:
            outcomes[-1] = "Success"
            n_success += 1

print('\n===========================================')
print (f"\nSuccess rate = {n_success/N*100}%")
print('Q-table after training:\n', qtable)

# Plot outcomes
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set_xlabel("Run Number")
ax.set_ylabel("Outcome")
ax.set_facecolor('#efeeea')
ax.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()