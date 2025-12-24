import gymnasium as gym
import torch
import numpy as np
import cv2
import tetris_gymnasium

# --- Load the Model ---
from main import TetrisDQN

def test():
    # 1. Create Environment in HUMAN mode
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")

    # 2. Match the state dimension from your training
    obs, _ = env.reset()
    state_dim = np.prod(obs['board'].shape)
    n_actions = env.action_space.n

    # 3. Initialize and Load Weights
    model = TetrisDQN(state_dim, n_actions)
    model.load_state_dict(torch.load("tetris_m1_model.pth", map_location="cpu"))
    model.eval()  # Set to evaluation mode (turns off dropout, etc.)

    print("Model loaded. Starting the game...")

    for episode in range(5):  # Watch 5 games
        obs, _ = env.reset()
        state = obs['board'].flatten()
        terminated = False
        truncated = False
        total_reward = 0

        env.render()

        while not (terminated or truncated):
            # NO random actions - only best moves
            env.render()
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = model(state_t).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = next_obs['board'].flatten()
            total_reward += reward

            # Render is handled automatically by gymnasium in 'human' mode
            # But we can add a tiny delay to make it watchable
            cv2.waitKey(50)

        # observation, reward, terminated, truncated, info = env.step(action)

        print(f"Game {episode + 1} Finished | Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    test()