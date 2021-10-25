import os
import gym
import math
from itertools import count

import random
from typing import List, NamedTuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from gymenv import MancalaEnv
from agent import Agent
from simple import MaxAgent

model_fn = os.path.join("save", "policy")
REPORTING_PERIOD = 100

if torch.cuda.is_available():

    # GPU Config

    device = torch.device('cuda')
    map_location = 'cuda'

    # Training settings

    BATCH_SIZE = 8
    GAMMA = 0.98
    EPS_START = 1
    EPS_END = 0.01
    EPS_DECAY = 0.0000025
    MEMORY_SIZE = 2000000
    LR = 0.0001
    UPDATE_TARGET = 4000

else:
    # CPU Config

    device = torch.device('cpu')
    map_location = 'cpu'

    # Training settings

    BATCH_SIZE = 8
    GAMMA = 0.98
    EPS_START = 1
    EPS_END = 0.01
    EPS_DECAY = 0.0000025
    MEMORY_SIZE = 2000000
    LR = 0.0001
    UPDATE_TARGET = 5000

ZEROED_ACTION_MASK = torch.zeros((BATCH_SIZE, 6)).to(device)


class MancalaAgentModel(nn.Module):

    def __init__(self):
        super(MancalaAgentModel, self).__init__()

        self.fc1 = nn.Linear(14, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 6)

    def forward(self, x, action_mask):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

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


class MaxQStrategy:
    def get_exploration_rate(self, current_step):
        return 0


class DeepQAgent(Agent):
    def __init__(self, strategy, device, policy_net = None):
        super().__init__()

        self.strategy = strategy
        self.device = device

        if policy_net is not None:
            self.policy_net = policy_net
        else:
            self.policy_net = MancalaAgentModel().to(device)

    def get_exploration_rate(self):
        return self.strategy.get_exploration_rate(self.current_step)

    def select_action(self, state, valid_actions, **kwargs):
        super().select_action(state, valid_actions)

        training_mode = False if "training_mode" not in kwargs else kwargs['training_mode']

        rate = self.get_exploration_rate()

        if rate > random.random():
            if len(valid_actions) == 0:
                return np.int64(0)
            else:
                return np.random.choice(valid_actions)
        else:

            if training_mode:
                action_mask = torch.zeros(6, dtype=torch.float).to(device)
            else:
                action_mask = torch.empty(6, dtype=torch.float).fill_(float("inf"), ).to(device)
                action_mask[valid_actions] = 0.

            action_mask = action_mask.unsqueeze(0)

            with torch.no_grad():
                input_t = torch.FloatTensor(state).unsqueeze(0).to(device)

                input_t = input_t / 72. # Normalize

                values = self.policy_net(input_t, action_mask).to(self.device)

                # print("Q Values: {}".format(values))

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

    t1 = (torch.FloatTensor(batch.state) / 72.).reshape(BATCH_SIZE, 14).to(device)
    t2 = torch.LongTensor(batch.action).to(device)
    t3 = torch.LongTensor(batch.reward).to(device)
    t4 = (torch.FloatTensor(batch.next_state) / 72.).reshape(BATCH_SIZE, 14).to(device)

    return t1, t2, t3, t4


if __name__ == '__main__':

    env = MancalaEnv(has_screen=False)

    strategy = EpsilonGreedyStrategy(EPS_START, EPS_END, EPS_DECAY)

    if model_fn is not None and os.path.isfile(model_fn):
        print("Resuming from checkpoint: {} ...".format(model_fn))
        policy_net = torch.load(os.path.join(os.getcwd(), model_fn), map_location=map_location)
    else:
        policy_net = MancalaAgentModel().to(device)

    agent = DeepQAgent(strategy, device, policy_net)
    maxagent = MaxAgent()
    policy_net = agent.policy_net

    memory = ReplayMem(MEMORY_SIZE)

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

        n_episodes += 1
        ep_reward_model = 0

        state = env.reset()
        player_1_last_state = None
        player_2_last_state = None

        for timestep in count():

            valid_actions = env.get_valid_actions()

            if env.active_player == 0:

                if player_1_last_state is not None:
                    memory.push(Experience(player_1_last_state, player_1_action, state, player_1_reward))

                player_1_action = agent.select_action(state, valid_actions, training_mode=False)
                next_state, player_1_reward, done, info = env.step(player_1_action)

                player_1_last_state = state.copy()

                ep_reward_model += player_1_reward

            else:

                if player_2_last_state is not None:

                    memory.push(Experience(
                        MancalaEnv.shift_view_p2(player_2_last_state),
                        player_2_action,
                        MancalaEnv.shift_view_p2(state),
                        player2_reward
                        )
                    )

                # Choose an action from player 2's perspective

                player_2_action = maxagent.select_action(MancalaEnv.shift_view_p2(state), valid_actions, env=env)
                next_state, player2_reward, done, info = env.step(player_2_action)

                player_2_last_state = state

            state = next_state

            # Training

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

            # Handle end of episode

                if done and n_episodes % REPORTING_PERIOD == 0:

                    print("Episode {} completed, last {} episodes avg. duration: {}, avg. reward: {}".format(
                        n_episodes,
                        REPORTING_PERIOD,
                        np.mean(episode_durations[-REPORTING_PERIOD:]),
                        np.mean(episode_rewards[-REPORTING_PERIOD:])
                    ))

                    # Report on training
                    print("Total batches trained: {}, replay mem size: {}".format(
                        n_batches_total,
                        len(memory),
                      )
                    )
                    print("Average training loss: {}, "
                          "Agent exploration rate: {}".format(
                        total_loss / n_batches_this_period,
                        agent.get_exploration_rate(),
                      )
                    )
                    # Tensorboard reporting

                    writer.add_scalar("Training loss", total_loss / n_batches_this_period, n_episodes)
                    writer.add_scalar("Exploration rate", agent.get_exploration_rate(), n_episodes)
                    writer.add_scalar("Episode duration", np.mean(np.mean(episode_durations[-REPORTING_PERIOD:])),
                                      n_episodes)
                    writer.add_scalar("Reward earned by model", ep_reward_model / REPORTING_PERIOD, n_episodes)

                    total_loss = 0
                    n_batches_this_period = 0

                writer.flush()

                if done:

                    if n_episodes % UPDATE_TARGET == 0:
                        print("Updating target net & saving checkpoint...")
                        target_net.load_state_dict(policy_net.state_dict())

                        if os.path.isfile(model_fn):
                            os.remove(model_fn)
                        torch.save(policy_net, model_fn)

                        optimizer = optim.Adam(params=policy_net.parameters(), lr=LR)

                    episode_durations.append(timestep)
                    episode_rewards.append(ep_reward_model)

                    break


