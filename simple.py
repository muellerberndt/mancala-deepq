from agent import Agent
from gymenv import MancalaEnv
import numpy as np

class MaxAgent(Agent):

    def __init__(self):
        super().__init__()

    def select_action(self, state, valid_actions) -> np.int64:
        env = MancalaEnv(has_screen=False)
        super().select_action(state, valid_actions)

        # Invalid State
        # Game should have ended already
        if len(valid_actions) == 0:
            return np.int64(0)
        
        # Get a score for all actions
        available_scores = list(
            map(lambda action:
                env.simulate_action(action),
                valid_actions))
        
        # Return the index of the first action with the highest score
        return np.argmax(available_scores)


class AdvancedMaxAgent(Agent):

    def __init__(self):
        super().init()

    def select_action(self, state, valid_actions) -> np.int64:
        super().select_action()