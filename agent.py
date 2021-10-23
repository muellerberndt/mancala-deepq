import numpy as np

class Agent(object):

    def __init__(self):
        self.current_step = 0

    def get_current_step(self) -> int:
        return self.current_step

    def reset_current_step(self, step):
        self.current_step = step

    def select_action(self, state, valid_actions) -> np.int64:
        self.current_step += 1