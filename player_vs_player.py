import os
import pygame
from pygame.locals import *

from gymenv import MancalaEnv, InvalidActionError

os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (30, 100)
os.environ['SDL_VIDEO_CENTERED'] = '0'

env = MancalaEnv(has_screen=True)

state = env.reset()

clock = pygame.time.Clock()

while 1:
    for event in pygame.event.get():

        env.render()

        if event.type == QUIT:
            pygame.quit()
        elif event.type == MOUSEBUTTONDOWN:

            action = env.get_index_from_coords(event.pos)

            if action > 0:
                try:
                    state, reward, done, info = env.step(action)
                except InvalidActionError:
                    print("Action is invalid")
                    pass

            print("Next active player: {}\nPlayer 1 score: {}\nPlayer 2 score: {}".format(
                env.get_active_player(),
                env.get_player_score(1),
                env.get_player_score(2)
            ))

        clock.tick(60)
