import os
import sys
import pygame
from pygame.locals import MOUSEBUTTONDOWN, QUIT
import torch

from gymenv import MancalaEnv, InvalidActionError, InvalidCoordinatesError
from deepq import MancalaAgentModel

model_fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join("save", "policy")
MODEL_SAVE_DIR = 'save'

if torch.cuda.is_available():

    # GPU Config

    device = torch.device('cuda')

else:
    # CPU Config

    device = torch.device('cpu')


os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (30, 100)
os.environ['SDL_VIDEO_CENTERED'] = '0'

''' 
TODO: Load the trained policy net
policy_net = torch.load(os.path.join(os.getcwd(), model_fn), map_location='cpu')
policy_net.eval()
'''

policy_net = MancalaAgentModel()

env = MancalaEnv(has_screen=True)


state = env.reset()

done = False

reward_earned = 0

env = MancalaEnv(has_screen=True)

state = env.reset()

clock = pygame.time.Clock()

while 1:

    env.render()

    if env.get_active_player() == 2:  # AI

        input_t = torch.FloatTensor(state[2]).unsqueeze(0).to(device)

        values = policy_net(input_t).to(device)

        actions = torch.topk(values, 3).indices[0]

        print("Actions topk: {}".format(actions))

        try:
            state, reward, done, info = env.step(actions[0])
        except InvalidActionError:
            print("Top action invalid, trying second best")
            try:
                state, reward, done, info = env.step(actions[1])
            except InvalidActionError:
                print("Second best action invalid, trying third best")
                try:
                    state, reward, done, info = env.step(actions[2])
                except InvalidActionError:
                    print("Top 3 actions invalid, resetting environment")
                    env.reset()

    else:
        for event in pygame.event.get():

            if event.type == QUIT:
                pygame.quit()
            elif event.type == MOUSEBUTTONDOWN:

                try:
                    action = env.get_action_from_coords(event.pos)
                except InvalidCoordinatesError:
                    continue

                try:
                    state, reward, done, info = env.step(action)
                except InvalidActionError:
                    print("Action is invalid")

                print("Next active player: {}\nPlayer 1 score: {}\nPlayer 2 score: {}".format(
                    env.get_active_player(),
                    env.get_player_score(1),
                    env.get_player_score(2)
                ))

            clock.tick(60)



