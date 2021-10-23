from agent import Agent
import numpy as np

class MaxAgent(Agent):

    def __init__(self):
        super().init()

    def select_action(self, state, valid_actions) -> np.int64:
        super.select_action()


class AdvancedMaxAgent(Agent):

    def __init__(self):
        super().init()

    def select_action(self, state, valid_actions) -> np.int64:
        super.select_action()