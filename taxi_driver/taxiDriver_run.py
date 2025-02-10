import gymnasium as gym
import numpy as np
import time

def main():
    q_table = np.load("q_table.npy")

    env = gym.make("Taxi-v3", render_mode="human")

    num_episodes = 2

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        print(f"\nIniciando o Episódio {episode}")
        time.sleep(1)

        while not done:
            env.render()
            action = np.argmax(q_table[state])
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            print(f"Ação: {action} | Recompensa: {reward}")
            time.sleep(1)
            if done or truncated:
                break

        print(f"Episódio {episode} finalizado com recompensa total: {total_reward}\n")
        time.sleep(0.01)

    env.close()

if __name__ == "__main__":
    main()
