import os
import sys
import time
import pygame
import torch
from simple import MaxAgent
import numpy as np

from gymenv import MancalaEnv
from deepq import MancalaAgentModel, DeepQAgent, MaxQStrategy

model_fn = os.path.join("save", "policy")


def debug_print(player, initial_state, env, action, reward):

    format_state = lambda s: "[{}] {} [{}] {}".format(s[0], s[1:7], s[7], s[8:14])

    print("{} action: {}\nState before:{}\nState after: {}\nReward: {}\n".format(
        player,
        action,
        format_state(initial_state),
        format_state(env.state),
        reward
    ))


def handle_game_end():
    if done:
        print("Game has ended!\nFinal scores: P1 {}, P2 {}".format(
            env.get_player_score(0),
            env.get_player_score(1)
              ))
        env.reset()
        env.reset()


os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (30, 100)
os.environ['SDL_VIDEO_CENTERED'] = '0'


if torch.cuda.is_available():

    # GPU Config

    device = torch.device('cuda')

else:
    # CPU Config

    device = torch.device('cpu')


env = MancalaEnv(has_screen=True)

if model_fn is not None and os.path.isfile(model_fn):
    print("Loading model: {} ...".format(model_fn))
    policy_net = torch.load(os.path.join(os.getcwd(), model_fn), map_location='cpu')
else:
    policy_net = MancalaAgentModel().to(device)


agent1 = DeepQAgent(MaxQStrategy(), device, policy_net=policy_net)
agent2 = MaxAgent()

state = env.reset()

done = False

clock = pygame.time.Clock()

while 1:

    env.render()

    valid_actions = env.get_valid_actions()

    if env.get_active_player() == 1:  # Player 2

        p2_view = MancalaEnv.shift_view_p2(state)

        action = agent1.select_action(p2_view, valid_actions)

        time.sleep(0.25)
        env.indicate_action_on_screen(action)
        time.sleep(1.0)

        state, reward, done, info = env.step(action)
        debug_print("Player 2:", p2_view, env, action, reward)

    else:

        action = agent2.select_action(state, valid_actions, env)

        time.sleep(0.25)
        env.indicate_action_on_screen(valid_actions[action])
        time.sleep(1.0)

        state, reward, done, info = env.step(valid_actions[action])

        debug_print("Player 1:", state, env, action, reward)

    if done:
        handle_game_end()
