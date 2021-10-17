import os
import gym
import math
from itertools import count
from gymenv import MancalaEnv

import random
from typing import List, NamedTuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

MODEL_SAVE_DIR = 'save'


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / float(obs.max())


if torch.cuda.is_available():

    # GPU Config

    device = torch.device('cuda')

    # World settings

    WORLD_SQUARE_LEN = 16
    STARTING_LIFE = 512
    MAX_APPLES = 1

    # Training settings

    BATCH_SIZE = 256
    GAMMA = 0.9
    EPS_START = 1
    EPS_END = 0.25
    EPS_DECAY = 0.000001
    MEMORY_SIZE = 5000000
    REWARD_GOAL = 1000
    LR = 0.00001
    TOTAL_STEPS = 100000000000
    UPDATE_TARGET = 1000

    KERNEL_SIZE = 3
    STRIDE = 1

else:
    # CPU Config

    device = torch.device('cpu')

    # World settings

    WORLD_SQUARE_LEN = 6
    STARTING_LIFE = 64
    MAX_APPLES = 1

    # Training settings

    BATCH_SIZE = 128
    GAMMA = 0.9
    EPS_START = 1
    EPS_END = 0.01
    EPS_DECAY = 0.0001
    MEMORY_SIZE = 2000000
    REWARD_GOAL = 1000
    LR = 0.00001
    TOTAL_STEPS = 100000000000
    UPDATE_TARGET = 1000

    KERNEL_SIZE = 2
    STRIDE = 1


class MancalaAgentModel(nn.Module):

    def __init__(self, world_shape):
        super(MancalaAgentModel, self).__init__()

        def conv2d_size_out(size, kernel_size=KERNEL_SIZE, stride=STRIDE):
            return (size - (kernel_size - 1) - 1) // stride + 1

        self.l1 = nn.Conv2d(1, 16, KERNEL_SIZE, stride=STRIDE)
        self.l2 = nn.Conv2d(16, 32, KERNEL_SIZE, stride=STRIDE)

        convw = conv2d_size_out(conv2d_size_out(world_shape[0]))
        convh = conv2d_size_out(conv2d_size_out(world_shape[1]))

        linear_input_size = convw * convh * 32

        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        flat = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(flat))
        x = self.fc2(x)

        return x


Experience = NamedTuple(
    "Experience", [
        ("state", np.array),
        ("action", int),
        ("next_state", np.array),
        ("reward", int)]
)

Episode = NamedTuple("Episode", [("experiences", List[Experience]), ("reward", int)])


class ReplayMem:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def __len__(self):
        return len(self.memory)

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy:

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)


class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def get_exploration_rate(self):
        return self.strategy.get_exploration_rate(self.current_step)

    def get_current_step(self):
        return self.current_step

    def reset_current_step(self, step):
        self.current_step = step

    def select_action(self, state, policy_net):
        rate = self.get_exploration_rate()
        self.current_step += 1

        if rate > random.random():
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                input_t = torch.DoubleTensor(state).unsqueeze(0).unsqueeze(0).to(device)

                values = policy_net(input_t).to(self.device)

                return values.argmax(dim=1)


class QValues:

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states, device):
        flattened = next_states.flatten(start_dim=1)
        _max = flattened.max(dim=1)

        _final_state_locations = _max[0].eq(0)
        final_state_locations = _final_state_locations.type(torch.bool)

        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.DoubleTensor(batch.state).reshape(BATCH_SIZE, 1, WORLD_SQUARE_LEN, WORLD_SQUARE_LEN).to(device)
    t2 = torch.LongTensor(batch.action).to(device)
    t3 = torch.LongTensor(batch.reward).to(device)
    t4 = torch.DoubleTensor(batch.next_state).reshape(BATCH_SIZE, 1, WORLD_SQUARE_LEN, WORLD_SQUARE_LEN).to(device)

    return t1, t2, t3, t4


if __name__ == '__main__':

    torch.set_default_dtype(torch.double)

    os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (30, 100)
    os.environ['SDL_VIDEO_CENTERED'] = '0'

    world_shape = (WORLD_SQUARE_LEN, WORLD_SQUARE_LEN)

    env = ScaledFloatFrame(MancalaEnv(world_shape, has_screen=False))
    strategy = EpsilonGreedyStrategy(EPS_START, EPS_END, EPS_DECAY)
    agent = Agent(strategy, 4, device)
    memory = ReplayMem(MEMORY_SIZE)

    policy_net = MancalaAgentModel(world_shape).to(device)
    target_net = MancalaAgentModel(world_shape).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(params=policy_net.parameters(), lr=LR)

    writer = SummaryWriter()

    iteration = 1

    n_batches_trained = 0

    episode_durations = []
    episode_rewards = []
    total_loss = 0
    step = 0
    steps = 0

    n_episodes = 0

    experiences_gathered = 0

    while 1:

        state = env.reset()

        n_episodes += 1
        ep_reward = 0

        saved_step = agent.get_current_step()

        for timestep in count():

            step += 1

            # env.render()

            # 0 = UP
            # 1 = DOWN
            # 2 = LEFT
            # 3 =  RIGHT

            action = agent.select_action(state, policy_net)

            next_state, reward, done, info = env.step(action)

            if not done:
                memory.push(Experience(state, action, next_state, reward))
            else:
                # Return zero state when episode ends
                memory.push(Experience(state, action, np.zeros(world_shape), reward))

            ep_reward += reward

            state = next_state

            if memory.can_provide_sample(BATCH_SIZE):
                experiences = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states, device)
                target_q_values = (next_q_values * GAMMA) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss
                n_batches_trained += 1
                steps += 1

                if step % UPDATE_TARGET == 0:
                    print("Updating target net & saving checkpoint...")
                    target_net.load_state_dict(policy_net.state_dict())

                    fn = os.path.join(os.getcwd(), MODEL_SAVE_DIR, "policy")

                    if os.path.isfile(fn):
                        os.remove(fn)
                    torch.save(policy_net, fn)

                if step > 0 and step % 100 == 0:
                    print("Avg. loss for training period: {}, "
                          "Agent exploration rate: {}".format(
                        total_loss / steps,
                        agent.get_exploration_rate(),
                        )
                    )

                    total_loss = 0
                    steps = 0

            if done or ep_reward >= REWARD_GOAL:

                print("Episode {} completed, steps: {}, reward: {}".format(
                    n_episodes,
                    timestep,
                    ep_reward
                ))

                print("Total training iterations: {}, replay mem size: {}".format(n_batches_trained, len(memory)))

                writer.add_scalar("Episode durations", timestep, n_episodes)
                writer.add_scalar("Exploration rate", agent.get_exploration_rate(), n_episodes)

                writer.flush()

                break

    print("Training goal reached")

    env.close()


