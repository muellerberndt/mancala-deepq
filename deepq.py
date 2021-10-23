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
REPORTING_PERIOD = 20


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 72.


if torch.cuda.is_available():

    # GPU Config

    device = torch.device('cuda')

    # Training settings

    BATCH_SIZE = 2048
    GAMMA = 0.98
    EPS_START = 1
    EPS_END = 0.25
    EPS_DECAY = 0.000005
    MEMORY_SIZE = 5000000
    LR = 0.00001
    UPDATE_TARGET = 2500

else:
    # CPU Config

    device = torch.device('cpu')

    # Training settings

    BATCH_SIZE = 2048
    GAMMA = 0.98
    EPS_START = 1
    EPS_END = 0.01
    EPS_DECAY = 0.00001
    MEMORY_SIZE = 2000000
    LR = 0.000001
    UPDATE_TARGET = 2500


ZEROED_ACTION_MASK = torch.zeros((BATCH_SIZE, 6))


class MancalaAgentModel(nn.Module):

    def __init__(self):
        super(MancalaAgentModel, self).__init__()

        self.fc1 = nn.Linear(14, 512)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, x, action_mask):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        result = torch.sub(x, action_mask)

        return result


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

    def select_action(self, state, policy_net, valid_actions):
        rate = self.get_exploration_rate()
        self.current_step += 1
        action_mask = torch.zeros((BATCH_SIZE, 6), dtype=float)

        if rate > random.random():
            if len(valid_actions) == 0:
                return np.int64(0)
            else:
                return np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                input_t = torch.FloatTensor(state).unsqueeze(0).to(device)

                values = policy_net(input_t, action_mask).to(self.device)

                return np.int64(torch.argmax(values).item())


class QValues:

    @staticmethod
    def get_current(policy_net, states, actions):
        _actions = actions.unsqueeze(-1)
        result = policy_net(states, ZEROED_ACTION_MASK)
        return result.gather(dim=1, index=_actions)

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
        action_mask = torch.zeros(non_final_states.shape[0], 6).to(device)
        values[non_final_state_locations] = target_net(non_final_states, action_mask).max(dim=1)[0].detach()
        return values


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.FloatTensor(batch.state).reshape(BATCH_SIZE, 14).to(device)
    t2 = torch.LongTensor(batch.action).to(device)
    t3 = torch.LongTensor(batch.reward).to(device)
    t4 = torch.FloatTensor(batch.next_state).reshape(BATCH_SIZE, 14).to(device)

    return t1, t2, t3, t4


if __name__ == '__main__':

    env = MancalaEnv(has_screen=False)

    strategy = EpsilonGreedyStrategy(EPS_START, EPS_END, EPS_DECAY)

    agent = Agent(strategy, 4, device)

    memory = ReplayMem(MEMORY_SIZE)

    policy_net = MancalaAgentModel().to(device)
    target_net = MancalaAgentModel().to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(params=policy_net.parameters(), lr=LR)

    writer = SummaryWriter()

    n_episodes = 0
    episode_durations = []
    episode_rewards = []
    total_loss = 0
    n_batches_total = 0
    n_batches_this_period = 0

    while 1:

        state = env.reset()

        n_episodes += 1
        ep_reward = 0

        for timestep in count():

            valid_actions = env.get_valid_actions()

            if env.active_player == 0:
                action = agent.select_action(state, policy_net, valid_actions)
                next_state, reward, done, info = env.step(action)
            else:
                state = MancalaEnv.shift_view_p2(state)
                action = agent.select_action(state, policy_net, valid_actions)
                next_state, reward, done, info = env.step(action)
                next_state = MancalaEnv.shift_view_p2(next_state)

            if not done:
                memory.push(Experience(state, action, next_state, reward))
            else:
                # Return zero state when episode ends
                memory.push(Experience(state, action, np.zeros(14), reward))

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
                n_batches_total += 1
                n_batches_this_period += 1

                if n_batches_total % UPDATE_TARGET == 0:
                    print("Updating target net & saving checkpoint...")
                    target_net.load_state_dict(policy_net.state_dict())

                    fn = os.path.join(os.getcwd(), MODEL_SAVE_DIR, "policy")

                    if os.path.isfile(fn):
                        os.remove(fn)
                    torch.save(policy_net, fn)

                if n_batches_total > 0 and n_batches_total % 200 == 0:
                    print("Total batches trained: {}, replay mem size: {}".format(
                        n_batches_total,
                        len(memory),
                        )
                    )
                    print("Avg. loss for last training period: {}, "
                          "Agent exploration rate: {}".format(
                        total_loss / n_batches_this_period,
                        agent.get_exploration_rate(),
                        )
                    )

                    total_loss = 0
                    n_batches_this_period = 0

            if done:
                episode_durations.append(timestep)
                episode_rewards.append(ep_reward)

                if n_episodes % REPORTING_PERIOD == 0:

                    print("Episode {} completed, last {} episodes avg. duration: {}, avg. reward: {}".format(
                        n_episodes,
                        REPORTING_PERIOD,
                        np.mean(episode_durations[-REPORTING_PERIOD:]),
                        np.mean(episode_rewards[-REPORTING_PERIOD:])
                    ))


                    writer.add_scalar("Exploration rate", agent.get_exploration_rate(), n_episodes)

                    writer.flush()

                break



