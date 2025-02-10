import matplotlib.pyplot as plt
import re
import numpy as np

def parse_training_log(log_file):
    episode_rewards = {}
    
    episode_pattern = re.compile(r"Episode:\s*(\d+)")
    reward_pattern  = re.compile(r"Reward:\s*(-?\d+\.?\d*)")
    
    with open(log_file, 'r') as f:
        for line in f:
            episode_match = episode_pattern.search(line)
            reward_match = reward_pattern.search(line)
            
            if episode_match and reward_match:
                try:
                    episode_num = int(episode_match.group(1))
                    reward = float(reward_match.group(1))
                    
                    if episode_num not in episode_rewards:
                        episode_rewards[episode_num] = reward
                    else:
                        episode_rewards[episode_num] += reward
                except Exception as e:
                    print(f"Erro ao processar a linha: {line}\nErro: {e}")

    return episode_rewards

def main():
    log_file = "training_log.txt"
    print("Iniciando processamento do log de treinamento...")
    episode_rewards = parse_training_log(log_file)
    
    if not episode_rewards:
        print("Nenhum dado encontrado no log de treinamento.")
        return

    episodes = sorted(episode_rewards.keys())
    rewards = [episode_rewards[ep] for ep in episodes]
    
    window_size = 50  
    rewards_smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    episodes_smoothed = episodes[window_size-1:] 

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, label="Recompensa Acumulada (Original)", color='gray', alpha=0.3)
    plt.plot(episodes_smoothed, rewards_smoothed, label=f"Recompensa Acumulada (Média Móvel, janela={window_size})",
             color='blue', linewidth=2)
    plt.xlabel("Número do Episódio")
    plt.ylabel("Recompensa Acumulada")
    plt.title("Curva de Aprendizado do Agente")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
