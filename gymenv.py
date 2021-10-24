from typing import Tuple

import numpy
import numpy as np
import math
import gym

class InvalidActionError(Exception):
    pass

class InvalidCoordinatesError(Exception):
    pass

PIXEL_WIDTH = 70
INVALID_ACTION_REWARD = -10
WINNER_REWARD = 0

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)

# This sets the margin between each cell
MARGIN = 1

import pygame
import copy

pygame.font.init()
myfont = pygame.font.SysFont('Arial', math.floor(PIXEL_WIDTH / 5))


class MancalaEnv(gym.Env):

    def __init__(self,
                 has_screen: bool = True
                 ):

        self.has_screen = has_screen
        self.active_player = 0

        self.action_space = gym.spaces.Discrete(6)

        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(1),                               # Player who has the next move
            gym.spaces.Box(0, 255, shape=(14,), dtype=np.long),   # Player 1 observation
            gym.spaces.Box(0, 255, shape=(14,), dtype=np.long)    # Player 2 observation
        ))

        self.state = np.zeros((14,), dtype=np.long)
        self.state[1:7] = 6
        self.state[9:14] = 6

        if has_screen:
            pygame.init()
            pygame.display.set_caption("Mancala")

            self.screen = pygame.display.set_mode([
                8 * (PIXEL_WIDTH + MARGIN),
                4 * (PIXEL_WIDTH + MARGIN)]
            )

    def advance_ptr(self, _ptr):

        ptr = 0 if _ptr == 13 else _ptr + 1

        if self.active_player == 1 and ptr == 7:
            ptr += 1

        if self.active_player == 0 and ptr == 0:
            ptr += 1

        return ptr

    def simulate_action(self, action: int):
        """
        An action is simulated to determine the best option of an action given a current state.
        The action is in the range between 0 and 5, excluding invalid actions e.g count 0 of an index.
        """

        if self.active_player == 0:
            index = action + 1
        else:
            index = 8 + action

        assert isinstance(self.state[0], numpy.int64)

        assert isinstance(action, numpy.int64)

        n_stones = self.state[index]

        state_copy = copy.copy(self.state)
        score = 0
        if n_stones == 0:
            return score

        state_copy[index] = 0

        ptr = index
        for i in range(n_stones):
            # Advance a "pointer" on the state array that follows certain rules.
            ptr = self.advance_ptr(ptr)
            state_copy[ptr] += 1

        '''
        Check if the move ended in an empty field. If it does the player wins all balls from
        the opposite field.
        '''

        if state_copy[ptr] == 1 and ptr not in [0, 7]:

            opposite_index = 14 - ptr

            if state_copy[opposite_index] > 0:

                if self.active_player == 0 and ptr < 7:
                    state_copy[7] += state_copy[opposite_index] + 1
                    state_copy[ptr] = state_copy[opposite_index] = 0
                    score = state_copy[7]
                elif ptr > 7:
                    state_copy[0] += state_copy[opposite_index] + 1
                    state_copy[ptr] = state_copy[opposite_index] = 0
                    score = state_copy[0]

        # Return score+1 if the turn ends in the own score board
        # to reward this action over equal score outcomes

        if self.active_player == 0 and ptr == 7:
            score += 1
        elif ptr == 0:
            score += 1

        return score

    def do_action(self, action: int):
        """
        The player chooses an action between 0 and 5.
        The action will affect different fields depending on which player's turn it is.
        """
        
        if self.active_player == 0:
            index = action + 1
        else:
            index = 8 + action

        assert isinstance(self.state[0], numpy.int64)

        assert isinstance(action, numpy.int64)

        n_stones = self.state[index]

        # print("[ENV] Selected field: {}\n[ENV] Number of stones in selected field: {}\n".format(index, n_stones))

        if n_stones == 0:
            raise InvalidActionError

        self.state[index] = 0

        ptr = index

        for i in range(n_stones):
            # Advance a "pointer" on the state array that follows certain rules.
            ptr = self.advance_ptr(ptr)
            self.state[ptr] += 1

        '''
        Check if the move ended in an empty field. If it does the player wins all balls from
        the opposite field.
        '''

        if self.state[ptr] == 1 and ptr not in [0, 7]:

            opposite_index = 14 - ptr

            if self.state[opposite_index] > 0:

                if self.active_player == 0 and ptr < 7:
                    self.state[7] += self.state[opposite_index] + 1
                    self.state[ptr] = self.state[opposite_index] = 0
                elif ptr > 7:
                    self.state[0] += self.state[opposite_index] + 1
                    self.state[ptr] = self.state[opposite_index] = 0

        # Switch active player except if the last move ended in the player's own "goal" field

        if self.active_player == 0 and ptr != 7:
            self.active_player = 1
        elif ptr != 0:
            self.active_player = 0

    def get_active_player(self) -> int:
        '''
        Return the player who has the next move. Return value is 0 (player 1) or 1 (player 2).
        '''

        return self.active_player

    def get_player_score(self, player: int) -> int:
        '''
        Return the player's current score. Argument is 0 (player 1) or 1 (player 2).
        '''
        return self.state[7] if player == 0 else self.state[0]

    def step(self, action: np.int64) -> np.array:

        done = False

        player = self.active_player
        initial_score = self.get_player_score(player)

        # End the game and return a negative reward if an invalid action is selected

        try:
            self.do_action(action)
        except InvalidActionError:
            # print("Game ended due to invalid action.")
            return self.get_observation(), INVALID_ACTION_REWARD, True, {}

        '''
        Check whether the game has ended.
        The game ends when one player has no longer any stones left.
        All remaining stones go to the player who made the last move.
        '''

        if np.sum(self.state[1:7]) == 0:
            # Player 1 is done

            for i in range(8, 14):
                self.state[0] += self.state[i]
                self.state[i] = 0

            done = True

        elif np.sum(self.state[8:14]) == 0:

            for i in range(1, 7):
                self.state[7] += self.state[i]
                self.state[i] = 0

            done = True

        elif self.get_player_score(0) > 36 or self.get_player_score(1) > 36:
            done = True

        reward = self.get_player_score(player) - initial_score

        return self.get_observation(), reward, done, {}

    def get_valid_actions(self) -> np.array:

        if self.active_player == 0:
            return np.where(self.state[1:7] != 0)[0]
        else:
            p2_view = self.shift_view_p2(self.state)[1:7]
            return np.where(p2_view != 0)[0]

    def get_observation(self) -> np.array:
        return np.copy(self.state)

    def reset(self):
        self.state = np.zeros((14,), dtype=np.long)

        self.state[1:7] = 6
        self.state[8:14] = 6

        self.active_player = 0

        return self.get_observation()

    def draw_board(self, player):

        if player == 0:
            state = self.state
            bg_color = WHITE if self.active_player == 0 else GRAY
        else:
            state = self.shift_view_p2(self.state)
            bg_color = WHITE if self.active_player == 1 else GRAY

        offset_y = ((PIXEL_WIDTH + MARGIN) * 2 + 2) * player

        # Draw the playing field

        pygame.draw.rect(self.screen,
                         bg_color,
                         [0,
                          offset_y,
                          PIXEL_WIDTH,
                          PIXEL_WIDTH * 2])

        textsurface = myfont.render("{}".format(state[0]), False, (0, 0, 0))

        self.screen.blit(textsurface, (PIXEL_WIDTH / 2, offset_y + PIXEL_WIDTH))

        for i in range(1, 7):
            textsurface = myfont.render("{}".format(state[i]), False, (0, 0, 0))

            pos_x = MARGIN + PIXEL_WIDTH + ((MARGIN + PIXEL_WIDTH) * ((i - 1)) + MARGIN)
            pos_y = offset_y + MARGIN + PIXEL_WIDTH

            pygame.draw.rect(self.screen,
                             bg_color,
                             [pos_x,
                              pos_y,
                              PIXEL_WIDTH,
                              PIXEL_WIDTH])

            self.screen.blit(textsurface, (pos_x + PIXEL_WIDTH / 2, pos_y + PIXEL_WIDTH / 2))

        pygame.draw.rect(self.screen,
                         bg_color,
                         [(MARGIN + PIXEL_WIDTH) * 7 + MARGIN,
                          offset_y,
                          PIXEL_WIDTH,
                          PIXEL_WIDTH * 2])

        textsurface = myfont.render("{}".format(state[7]), False, (0, 0, 0))

        self.screen.blit(textsurface, (PIXEL_WIDTH * 7 + PIXEL_WIDTH / 2, offset_y + PIXEL_WIDTH))

        for i in range(8, 14):
            textsurface = myfont.render("{}".format(state[i]), False, (0, 0, 0))

            pos_x = MARGIN + PIXEL_WIDTH + ((MARGIN + PIXEL_WIDTH) * ((13 - i)) + MARGIN)
            pos_y = offset_y

            pygame.draw.rect(self.screen,
                             bg_color,
                             [pos_x,
                              pos_y,
                              PIXEL_WIDTH,
                              PIXEL_WIDTH])

            self.screen.blit(textsurface, (pos_x + PIXEL_WIDTH / 2, pos_y + PIXEL_WIDTH / 2))

    def indicate_action_on_screen(self, action):

        pos_y = (PIXEL_WIDTH * MARGIN) * 3 if self.active_player == 1 else PIXEL_WIDTH * MARGIN
        pos_x = PIXEL_WIDTH * MARGIN + (PIXEL_WIDTH * MARGIN) * action

        pygame.draw.rect(self.screen,
                         RED,
                         [pos_x,
                          pos_y,
                          PIXEL_WIDTH,
                          PIXEL_WIDTH],
                         3)

        pygame.display.flip()


    def render(self, mode="human"):

        if self.has_screen:

            self.screen.fill(DARK_GRAY)

            self.draw_board(0)
            self.draw_board(1)

            pygame.display.flip()

    def get_action_from_coords(self, pos: Tuple) -> np.int64:
        '''
        Determine the field clicked by the player from pixel coordinates.
        Returns the state index selected if the coordinates are within the active player's
        screen area or 0 otherwise.
        '''

        if self.active_player == 0 and PIXEL_WIDTH < pos[1] < PIXEL_WIDTH * 2 \
                and MARGIN + PIXEL_WIDTH < pos[0] < (MARGIN + PIXEL_WIDTH) * 7:
            return np.int64(math.floor(pos[0] / PIXEL_WIDTH) - 1)

        if self.active_player == 1 and 3 * PIXEL_WIDTH < pos[1] < 4 * PIXEL_WIDTH \
                and MARGIN + PIXEL_WIDTH < pos[0] < (MARGIN + PIXEL_WIDTH) * 7:
            return np.int64(math.floor(pos[0] / PIXEL_WIDTH) - 1)

        raise InvalidCoordinatesError

    @staticmethod
    def shift_view_p2(state) -> np.array:
        """Returns a view of the state
        from the perspective of player 2 (basically
        shift the state array by half)
        """

        return numpy.append(state[7:14], state[0:7])