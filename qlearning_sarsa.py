import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Configuração do ambiente Taxi-Driver
env = gym.make('Taxi-v3').env

# Configuração do ambiente Cliff Walking
env_cliff = gym.make('CliffWalking-v0').env

# Parâmetros comuns para ambos os algoritmos
alpha = 0.1
gamma = 0.99
epsilon = 0.1
epsilon_min = 0.1
epsilon_dec = 1
episodes = 5000

# Função para treinar o agente com Q-Learning
def q_learning(env, alpha, gamma, epsilon, episodes):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            next_state, reward, done, _, _ = env.step(action)
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        epsilon = max(epsilon * epsilon_dec, epsilon_min)
    return q_table, rewards

# Função para treinar o agente com SARSA
def sarsa(env, alpha, gamma, epsilon, episodes):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * q_table[next_state, next_action] - q_table[state, action]
            )
            state, action = next_state, next_action
            total_reward += reward
        rewards.append(total_reward)
        epsilon = max(epsilon * epsilon_dec, epsilon_min)
    return q_table, rewards

# Treinamento e comparação no ambiente Taxi-Driver
print("Treinando o agente no ambiente Taxi-Driver com Q-Learning...")
q_table_qlearning, rewards_qlearning = q_learning(env, alpha, gamma, epsilon, episodes)
print("Treinando o agente no ambiente Taxi-Driver com SARSA...")
q_table_sarsa, rewards_sarsa = sarsa(env, alpha, gamma, epsilon, episodes)

# Comparação das curvas de aprendizado
plt.plot(rewards_qlearning, label='Q-Learning')
plt.plot(rewards_sarsa, label='SARSA')
plt.xlabel('Episódios')
plt.ylabel('Recompensa')
plt.title('Curva de Aprendizado - Taxi-Driver')
plt.legend()
plt.show()

# Treinamento e visualização no ambiente Cliff Walking
print("Treinando o agente no ambiente Cliff Walking com Q-Learning...")
q_table_qlearning_cliff, _ = q_learning(env_cliff, alpha, gamma, epsilon, episodes)
print("Treinando o agente no ambiente Cliff Walking com SARSA...")
q_table_sarsa_cliff, _ = sarsa(env_cliff, alpha, gamma, epsilon, episodes)

# Animação do agente no Cliff Walking
env_cliff = gym.make("CliffWalking-v0", render_mode="human").env
(state, _) = env_cliff.reset()
done = False
while not done:
    action = np.argmax(q_table_qlearning_cliff[state])
    state, _, done, _, _ = env_cliff.step(action)

print("Notebook atualizado com treinamento, comparação e visualização do comportamento do agente!")
