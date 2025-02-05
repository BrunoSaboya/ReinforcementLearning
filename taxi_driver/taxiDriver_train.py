import gymnasium as gym
import numpy as np
import sys
import logging
env = gym.make("Taxi-v3")

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1
episodes = 10000
max_steps = 100

state_size = env.observation_space.n
action_size = env.action_space.n

# Set up logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO)

# Initialize Q-table
q_table = np.zeros((state_size, action_size))

# Training loop
for episode in range(episodes):
    state = env.reset()[0]
    done = False
    for _ in range(max_steps):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, _, _ = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Q-learning formula
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        # Log the experience
        logging.info(f"Episode: {episode}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

        state = next_state

        if done:
            break

# Save the Q-table
np.save("q_table.npy", q_table)