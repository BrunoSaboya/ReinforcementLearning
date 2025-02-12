import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train_q_learning(alpha, epsilon, gamma, episodes=1000, max_steps=100, seed=None):
    env = gym.make("Taxi-v3")
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.zeros((state_size, action_size))
    rewards_all = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        for step in range(max_steps):
            # Se o número aleatório for menor que epsilon, escolhe ação aleatória,
            # caso contrário, escolhe a ação com maior valor na Q-table.
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
                
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Atualização da Q-table:
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            
            total_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        rewards_all.append(total_reward)
    env.close()
    return rewards_all, q_table

def run_experiments_alpha(alpha_values, fixed_epsilon, gamma, episodes, max_steps, runs):
    results = {}
    for alpha in alpha_values:
        run_rewards = []
        for run in range(runs):
            rewards, _ = train_q_learning(alpha, fixed_epsilon, gamma, episodes, max_steps, seed=run)
            run_rewards.append(rewards)
        results[alpha] = run_rewards
    return results

def run_experiments_epsilon(epsilon_values, fixed_alpha, gamma, episodes, max_steps, runs):
    results = {}
    for epsilon in epsilon_values:
        run_rewards = []
        for run in range(runs):
            rewards, _ = train_q_learning(fixed_alpha, epsilon, gamma, episodes, max_steps, seed=run)
            run_rewards.append(rewards)
        results[epsilon] = run_rewards
    return results

def plot_all_runs(results, title, xlabel="Episódio", ylabel="Recompensa Acumulada"):
    plt.figure(figsize=(12, 6))
    color_map = plt.cm.get_cmap("tab10")
    param_values = sorted(results.keys())
    
    for idx, param_value in enumerate(param_values):
        runs_rewards = np.array(results[param_value])
        episodes = np.arange(runs_rewards.shape[1])
        
        # Escolhe uma cor para esse parâmetro
        color = color_map(idx % 5)
        # Plota cada run com transparência
        for i, run_rewards in enumerate(runs_rewards):
            plt.plot(episodes, run_rewards, color=color, alpha=0.15,
                     label=f"Valor = {param_value}" if i == 0 else None)
        # Plota a média entre as runs para o parâmetro (linha mais espessa)
        avg_rewards = runs_rewards.mean(axis=0)
        plt.plot(episodes, avg_rewards, color=color, linewidth=2,
                 label=f"Valor = {param_value} - Média")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Parâmetros gerais
    episodes = 1000       # Número de episódios de treinamento
    max_steps = 100       # Número máximo de passos por episódio
    runs = 5              # Número de runs para cada configuração
    gamma = 0.6           # Fator de desconto

    # ===============================
    # Experimento 1: Variação de α (alpha) com ε fixo
    # ===============================
    alpha_values = [0.01, 0.1, 0.9]   # Testa alpha muito baixo, intermediário e muito alto
    fixed_epsilon = 0.1               # Valor fixo para epsilon
    print("Executando experimentos variando α (com ε fixo = 0.1)...")
    results_alpha = run_experiments_alpha(alpha_values, fixed_epsilon, gamma, episodes, max_steps, runs)
    plot_all_runs(results_alpha, title=f"Curva de Aprendizado Variando α (ε fixo = {fixed_epsilon})")

    # ===============================
    # Experimento 2: Variação de ε (epsilon) com α fixo
    # ===============================
    epsilon_values = [0.01, 0.1, 0.9]   # Testa epsilon muito baixo, intermediário e muito alto
    fixed_alpha = 0.1                   # Valor fixo para alpha
    print("Executando experimentos variando ε (com α fixo = 0.1)...")
    results_epsilon = run_experiments_epsilon(epsilon_values, fixed_alpha, gamma, episodes, max_steps, runs)
    plot_all_runs(results_epsilon, title=f"Curva de Aprendizado Variando ε (α fixo = {fixed_alpha})")

    # ===============================
    # Experimento 3: Impacto dos casos extremos de ε
    #           (a) ε = 1: Sempre aleatório (nunca consulta a Q-table)
    #           (b) ε = 0: Sempre greedy (sempre consulta a Q-table)
    # ===============================
    extreme_epsilons = [0.0, 1.0]
    print("Executando experimentos extremos: ε = 0 (sempre greedy) vs ε = 1 (sempre aleatório)...")
    results_extreme = run_experiments_epsilon(extreme_epsilons, fixed_alpha, gamma, episodes, max_steps, runs)
    plot_all_runs(results_extreme, title=f"Curva de Aprendizado Extremas: ε = 0 vs ε = 1 (α fixo = {fixed_alpha})")

if __name__ == "__main__":
    main()
