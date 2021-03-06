from agent import Agent
from gymenv import MancalaEnv
import numpy as np


class RandomAgent(Agent):

    def __init__(self):
        super().__init__()

    def select_action(self, state, valid_actions, **kwargs) -> np.int64:
        super().select_action(state, valid_actions)

        return np.random.choice(valid_actions)

class MaxAgent(Agent):

    def __init__(self):
        super().__init__()

    def select_action(self, state, valid_actions, **kwargs) -> np.int64:
        super().select_action(state, valid_actions)

        # Invalid State
        # Game should have ended already
        if len(valid_actions) == 0:
            return np.int64(0)
        
        # Get a score for all actions
        available_scores = list(
            map(lambda action:
                kwargs['env'].simulate_action(action),
                valid_actions))
        # print("MaxAgent action scores: {}".format(available_scores))
        # Return the index of the first action with the highest score
        best_action = valid_actions[np.argmax(available_scores)]
        # print("MaxAgent best action: {}\n".format(best_action))
        return best_action


class AdvancedMaxAgent(Agent):

    def __init__(self):
        super().init()

    def select_action(self, state, valid_actions, **kwargs) -> np.int64:
        super().select_action()