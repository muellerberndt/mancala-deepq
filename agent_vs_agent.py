import os
import sys
import time
import pygame
import torch
from simple import MaxAgent
import numpy as np
import random

from gymenv import MancalaEnv
from deepq import MancalaAgentModel, DeepQAgent, MaxQStrategy

'''
Sometimes select a random action
0 -> always follow agent policy
1 -> fully random
'''

RANDOMIZE_ACTIONS_RATE = 0


def debug_print(player, agent_class, state_before, state_after, action, reward):

    format_state = lambda s: "[{}] {}\n{} [{}]".format(s[0], np.flip(s[8:14]), s[1:7], s[7])

    print("-- {} - {}: \nState before:\n{}\nAction: {}\nState after:\n{}\nReward: {}\n".format(
        player,
        agent_class.__name__,
        format_state(state_before),
        action,
        format_state(state_after),
        reward
    ))


def handle_game_end():
    print("\n### Game has ended!\n### \nFinal scores: P1 {}, P2 {}".format(
        env.get_player_score(0),
        env.get_player_score(1)
          ))
    env.reset()


def display_action(action) -> bool:
    time.sleep(0.5)
    env.indicate_action_on_screen(action)
    time.sleep(0.5)

os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (30, 100)
os.environ['SDL_VIDEO_CENTERED'] = '0'


if torch.cuda.is_available():

    # GPU Config

    device = torch.device('cuda')

else:
    # CPU Config

    device = torch.device('cpu')


env = MancalaEnv(has_screen=True)

model_fn = os.path.join(os.getcwd(), "save", "policy")
policy_net = torch.load(model_fn, map_location='cpu')

player1 = DeepQAgent(MaxQStrategy(), device, policy_net=policy_net)
player2 = MaxAgent()

clock = pygame.time.Clock()

while 1:

    state = env.reset()

    while 1:

        env.render()

        valid_actions = env.get_valid_actions()

        if env.get_active_player() == 0:  # Player 1

            old_state = state

            if random.random() < RANDOMIZE_ACTIONS_RATE:
                action = random.choice(valid_actions)
            else:
                action = player1.select_action(state, valid_actions, env=env, debug_q_values=True)

            display_action(action)

            state, reward, done, info = env.step(action)

            debug_print("Player 1:", type(player1), old_state, state, action, reward)

        else:  # Player 2

            p2_view = MancalaEnv.shift_view_p2(state)

            if random.random() < RANDOMIZE_ACTIONS_RATE:
                action = random.choice(valid_actions)
            else:
                action = player2.select_action(p2_view, valid_actions, env=env, debug_q_values=True)

            display_action(action)

            state, reward, done, info = env.step(action)

            debug_print("Player 2:", type(player2), p2_view, MancalaEnv.shift_view_p2(state), action, reward)

        if done:
            handle_game_end()

            break
