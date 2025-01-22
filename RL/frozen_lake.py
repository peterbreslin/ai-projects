import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

"""
Implementing the Q-Learning algorithm to solve the non-slippery FrozenLake environment from the Gymnasium library.

This basic tabular reinforcement learning approach trains the agent with the Q-Learning algorithm, which uses the Bellman Equation to determine and update the Q-values. The agent's performance is then evaluated post-training. Finally, the agent's trajectory is visualized in the environment in either a text-based representation as an ASCII-art grid(render_mode="ansi"), or a graphical representation where each cell of the lake is shown visually (render_mode="rgb_array").
"""


class FrozenLakeQLearning:
    def __init__(self, render_mode="ansi", N=1000, alpha=0.5, gamma=0.9):
        self.env = gym.make(
            "FrozenLake-v1", 
            desc=None, 
            map_name="4x4", 
            is_slippery=False, 
            render_mode=render_mode
        )
        self.qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.N = N
        self.alpha = alpha
        self.gamma = gamma
        self.render_mode = render_mode


    def train(self):
        for episode in range(self.N):
            state, info = self.env.reset()

            terminated = False
            while not terminated:
                # Choose the action with the highest value in the current state
                if np.max(self.qtable[state]) > 0:
                    action = np.argmax(self.qtable[state])
                else:
                # Random action if the Q-value is 0
                    action = self.env.action_space.sample()

                # Implement action and move agent
                new_state, reward, terminated, truncated, info = self.env.step(action)

                # Update Q(s,a) by the Bellman Equation
                self.qtable[state, action] = self.qtable[state, action] + self.alpha * (reward + self.gamma * np.max(self.qtable[new_state]) - self.qtable[state, action])

                # Update the current state
                state = new_state


    def evaluate(self, episodes=100):
        successs_count = 0
        for i in range(episodes):
            state, info = self.env.reset()
            terminated = False

            while not terminated:
                if np.max(self.qtable[state]) > 0:
                    action = np.argmax(self.qtable[state])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, terminated, truncated, info = self.env.step(action)
                state = new_state

                # Count successes
                if terminated and reward == 1.0:
                    successs_count += 1

        print (f"Success rate = {(successs_count/episodes)*100}%")


    def visualize_trajectory(self):
        state, info = self.env.reset()

        if self.render_mode == "rgb_array":
            frame = self.env.render()
            plt.imshow(frame)
            plt.axis("off")
            plt.pause(0.5)
            plt.clf()
        else:
            print(self.env.render())
            time.sleep(0.5)

        sequence = []
        terminated = False
        while not terminated:
            if np.max(self.qtable[state]) > 0:
                action = np.argmax(self.qtable[state])
            else:
                action = self.env.action_space.sample()

            sequence.append(action)
            new_state, reward, terminated, truncated, info = self.env.step(action)
            state = new_state

            if self.render_mode == "rgb_array":
                frame = self.env.render()
                plt.imshow(frame)
                plt.axis("off")
                plt.pause(0.5)
                plt.clf()
            else:
                print(self.env.render())
                time.sleep(0.5)

        print(f"Sequence = {sequence}")

        
if __name__ == "__main__":

    # Initialize the Q-Learning agent
    render_mode = "ansi"
    # render_mode = "rgb_array"
    agent = FrozenLakeQLearning(render_mode = render_mode)

    # Train the agent
    agent.train()

    # Evaluation
    agent.evaluate()

    # Visualize the trajectory
    agent.visualize_trajectory()