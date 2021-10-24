from agent import Agent
from gymenv import MancalaEnv
import numpy as np

class MaxAgent(Agent):

    def __init__(self):
        super().__init__()

    def select_action(self, state, valid_actions, env) -> np.int64:
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
        print("MaxAgent action scores: {}".format(available_scores))
        # Return the index of the first action with the highest score
        best_action = np.argmax(available_scores)
        print("MaxAgent best action: {}\n".format(best_action))
        return best_action


class AdvancedMaxAgent(Agent):

    def __init__(self):
        super().init()

    def select_action(self, state, valid_actions) -> np.int64:
        super().select_action()