import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
import torch
import cv2

# ----- verifying GPU acceleration -----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ----- creating the tetris enviroment -----
env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
env.reset(seed=42)

def run_test_game():
    observation, info = env.reset()
    terminated = False
    truncated = False

    print("Starting Tetris test engine...")

    while not (terminated or truncated):
        # ----- sampling a random action -----
        action = env.action_space.sample()

        # ----- step in the engine -----
        observation, reward, terminated, truncated, info = env.step(action)

        # ----- rendering the game window -----
        env.render()

        # ----- slowing it down for better visualization -----
        cv2.waitKey(200)

    print("Tetris test engine finished.")
    env.close()

if __name__ == "__main__":
    run_test_game()