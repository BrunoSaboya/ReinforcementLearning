import gymnasium as gym
import numpy as np
import time

def main():
    # Carrega a Q-table treinada
    q_table = np.load("q_table.npy")

    # Cria o ambiente do Taxi-v3 com renderização para visualização humana
    env = gym.make("Taxi-v3", render_mode="human")

    # Define o número de episódios de teste
    test_episodes = 5

    for episode in range(1, test_episodes + 1):
        # Inicializa o ambiente; observe que, no Gymnasium, reset() retorna (obs, info)
        state, _ = env.reset()
        done = False

        print(f"\nIniciando o Episódio {episode}")
        time.sleep(1)

        while not done:
            # Renderiza o ambiente (a janela gráfica do Gym será atualizada)
            env.render()
            # Seleciona a melhor ação para o estado atual com base na Q-table
            action = np.argmax(q_table[state])
            
            # Executa a ação no ambiente
            state, reward, done, truncated, _ = env.step(action)
            print(f"Ação escolhida: {action} | Recompensa recebida: {reward}")
            time.sleep(0.01)
            
            # Verifica se o episódio terminou (done ou truncated)
            if done or truncated:
                break

        print(f"Episódio {episode} finalizado.\n")
        time.sleep(2)

    env.close()

if __name__ == "__main__":
    main()
