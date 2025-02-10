import gymnasium as gym
import numpy as np
import sys
import logging
env = gym.make("Taxi-v3")

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 1
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 5000
max_steps = 100

state_size = env.observation_space.n
action_size = env.action_space.n

logging.basicConfig(filename='training_log.txt', level=logging.INFO)

q_table = np.zeros((state_size, action_size))

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    for _ in range(max_steps):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()

        else:
            action = np.argmax(q_table[state])  

        next_state, reward, done, _, _ = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        logging.info(f"Episode: {episode}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

        state = next_state

        if done:
            break
    epsilon = max(epsilon_min, epsilon*epsilon_decay)

np.save("q_table.npy", q_table)