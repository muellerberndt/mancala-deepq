import os
import torch
from simple import MaxAgent, RandomAgent
import numpy as np

from gymenv import MancalaEnv
from deepq import MancalaAgentModel, DeepQAgent, MaxQStrategy

if torch.cuda.is_available():
    location = 'cuda'
    device = torch.device('cuda')
else:
    location = 'cpu'
    device = torch.device('cpu')

model_fn = os.path.join(os.getcwd(), "save", "policy")
policy_net = torch.load(model_fn, map_location=location)


## Settings ##

NUM_GAMES = 1000
# player2 = RandomAgent()
player1 = RandomAgent()
player2 = DeepQAgent(MaxQStrategy(), device, policy_net=policy_net)

env = MancalaEnv(has_screen=False)

done = False

player1_wins = 0
player2_wins = 0

for i in range(1, NUM_GAMES):

    state = env.reset()

    for j in range(1, 3):
        valid_actions = env.get_valid_actions()

        action = np.random.choice(valid_actions)

        state, reward, done, info = env.step(action)

    while 1:

        valid_actions = env.get_valid_actions()

        if env.get_active_player() == 0:  # Player 1

            action = player1.select_action(state, valid_actions, env=env, debug_q_values=False)

            state, reward, done, info = env.step(action)

        else:  # Player 2

            action = player2.select_action(MancalaEnv.shift_view_p2(state), valid_actions, env=env, debug_q_values=False)

            state, reward, done, info = env.step(action)

        if done:
            if env.get_player_score(0) > env.get_player_score(1):
                player1_wins += 1
            elif env.get_player_score(1) > env.get_player_score(0):
                player2_wins += 1

            break

print("Win percentages: {} (player 1) {:.2f}%, {} (player 2) {:.2f}%, Draw {:.2f}%".format(
    type(player1).__name__,
    100 * float(player1_wins) / NUM_GAMES,
    type(player2).__name__,
    100 * float(player2_wins) / NUM_GAMES,
    100 * (float(1) - (player1_wins + player2_wins) / NUM_GAMES)
    )
)