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


def debug_print(env, state):
    print("Next active player: {}\nValid_actions: {}]\nPlayer 1 score: {}\nPlayer 2 score: {}\n\n".format(
        env.get_active_player(),
        state[3],
        env.get_player_score(1),
        env.get_player_score(2)
    ))

def handle_game_end(done: bool):
    if done:
        print("Game has ended.")
        env.reset()

action_mask = torch.ones(6, dtype=torch.float).unsqueeze(0)

while 1:

    env.render()

    if env.get_active_player() == 2:  # AI

        input_t = torch.FloatTensor(state[2]).unsqueeze(0).to(device)

        values = policy_net(input_t, action_mask).to(device)

        action = torch.argmax(values)

        state, reward, done, info = env.step(action)
        debug_print(env, state)

        handle_game_end(done)

    else:
        for event in pygame.event.get():

            if event.type == QUIT:
                pygame.quit()
            elif event.type == MOUSEBUTTONDOWN:

                try:
                    action = env.get_action_from_coords(event.pos)
                except InvalidCoordinatesError:
                    continue

                state, reward, done, info = env.step(action)
                debug_print(env, state)

                handle_game_end(done)




            clock.tick(60)



