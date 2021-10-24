from agent import Agent
from gymenv import MancalaEnv
import numpy as np

class MaxAgent(Agent):

    def __init__(self):
        super().init()

    def select_action(self, state, valid_actions) -> np.int64:
        env = MancalaEnv(has_screen=False)
        super().select_action(state, valid_actions)

        # Invalid State
        # Game should have ended already
        if len(valid_actions) == 0:
            return np.int64(0)

        # simulate an action for all valid actions
        # select the best action to play

class AdvancedMaxAgent(Agent):

    def __init__(self):
        super().init()

    def select_action(self, state, valid_actions) -> np.int64:
        super().select_action()