import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
import cv2
import torch

class TetrisDQN(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(TetrisDQN, self).__init__()

        # ----- input_shape can be either a flatened vector or a 2D grid -----
        self.fc1 = torch.nn.Linear(input_shape, 256)
        self.fc2 = torch.nn.Linear(256, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.head = torch.nn.Linear(256, n_actions)

    def forward(self, x):
        # ----- ensuring that the input is flattened -----
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))

        return self.head(x)

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
        cv2.waitKey(50)

    print("Tetris test engine finished.")
    env.close()

if __name__ == "__main__":
    # ----- verifying GPU acceleration -----
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- creating the tetris enviroment -----
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)

    model = TetrisDQN(200, env.action_space.n).to(device)
    targetModel = TetrisDQN(200, env.action_space.n).to(device)
    targetModel.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    run_test_game()