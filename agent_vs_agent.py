import os
import sys
import time
import pygame
import torch
from simple import MaxAgent, RandomAgent
import numpy as np

from gymenv import MancalaEnv
from deepq import MancalaAgentModel, DeepQAgent, MaxQStrategy


def debug_print(player, agent_class, state_before, state_after, action):

    format_state = lambda s: "[{}] {} [{}] {}".format(s[0], s[1:7], s[7], s[8:14])

    print("{} {} action: {}\nState before:{}\nState after: {}\n".format(
        player,
        agent_class.__name__,
        action,
        format_state(state_before),
        format_state(state_after),
    ))


def handle_game_end():
    print("Game has ended!\nFinal scores: P1 {}, P2 {}".format(
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

state = env.reset()

done = False

clock = pygame.time.Clock()

while 1:

    env.render()

    valid_actions = env.get_valid_actions()

    if env.get_active_player() == 0:  # Player 1

        old_state = state

        action = player1.select_action(state, valid_actions, env=env, debug_q_values=True)
        display_action(action)

        state, reward, done, info = env.step(action)

        debug_print("Player 1:", type(player1), old_state, state, action)

    else:  # Player 2

        p2_view = MancalaEnv.shift_view_p2(state)

        action = player2.select_action(p2_view, valid_actions, env=env, debug_q_values=True)
        display_action(action)

        state, reward, done, info = env.step(action)

        debug_print("Player 2:", type(player2), p2_view, MancalaEnv.shift_view_p2(state), action)

    if done:
        handle_game_end()
