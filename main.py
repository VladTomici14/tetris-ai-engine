import gymnasium as gym
from tensorflow.python.keras.mixed_precision.policy import policy_scope
from tetris_gymnasium.envs.tetris import Tetris
import cv2
import torch
from collections import deque
import random
import numpy as np

# TODO: add statistics and plots somehow for the results evaluation
# TODO: create video outputs of every model running


# ----- configuration and GPU parameters -----
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.9995
TARGET_UPDATE = 10
MEMORY_SIZE = 20000
LR = 1e-4

# ----- declaring the CNN architecture -----
class TetrisDQN(torch.nn.Module):
    def __init__(self, input_dim, n_actions):
        super(TetrisDQN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.FloatTensor(np.array(state)).to(DEVICE),
            torch.LongTensor(action).to(DEVICE),
            torch.FloatTensor(reward).to(DEVICE),
            torch.FloatTensor(np.array(next_state)).to(DEVICE),
            torch.BoolTensor(done).to(DEVICE)
        )


def train():
    # ----- creating the tetris enviroment -----
    env = gym.make("tetris_gymnasium/Tetris", render_mode=None)

    n_actions = env.action_space.n

    # ----- finding the state dimension based on the board shape (20x10 = 200) -----
    state_dim = np.prod(env.observation_space['board'].shape)

    # ----- loading the networks -----
    policy_net = TetrisDQN(state_dim, n_actions).to(DEVICE)
    target_net = TetrisDQN(state_dim, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    # ----- adding an optimizer -----
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    epsilon = EPS_START

    # ----- training the network -----
    for episode in range(1000):
        state, _ = env.reset()
        state = state['board'].flatten()
        total_reward = 0

        # ----- going through all the steps in one game -----
        for t in range(2000):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action = policy_net(state_t).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state['board'].flatten()
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)

            state = next_state

            total_reward += reward

            # ----- performing optimisation step -----
            if len(memory.buffer) > BATCH_SIZE:
                b_state, b_action, b_reward, b_next_state, b_done = memory.sample(BATCH_SIZE)

                # ----- current q values -----
                current_q = policy_net(b_state).gather(1, b_action.unsqueeze(1))

                # ----- target q values -----
                with torch.no_grad():
                    target_q_values = target_net(b_next_state)
                    max_next_q = target_q_values.max(1)[0]
                    expected_q = b_reward + (GAMMA * max_next_q * (~b_done))

                loss = torch.nn.MSELoss()(current_q.squeeze(), expected_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # ----- decay exploration -----
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # ----- update target network -----
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {episode} | Reward: {total_reward:.2f} | Eps: {epsilon:.2f}")

    torch.save(policy_net.state_dict(), "tetris_m1_model.pth")
    print("Training Complete. Model Saved.")


# def run_test_game():
#     observation, info = env.reset()
#     terminated = False
#     truncated = False
#
#     print("Starting Tetris test engine...")
#
#     while not (terminated or truncated):
#         # ----- sampling a random action -----
#         action = env.action_space.sample()
#
#         # ----- step in the engine -----
#         observation, reward, terminated, truncated, info = env.step(action)
#
#         # ----- rendering the game window -----
#         env.render()
#
#         # ----- slowing it down for better visualization -----
#         cv2.waitKey(50)
#
#     print("Tetris test engine finished.")
#     env.close()


if __name__ == "__main__":
    # model = TetrisDQN(200, env.action_space.n).to(device)
    # targetModel = TetrisDQN(200, env.action_space.n).to(device)
    # targetModel.load_state_dict(model.state_dict())
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #
    # run_test_game()

    train()
