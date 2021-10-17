import os
import sys
import time
import random
import torch

from gymenv import MancalaEnv
from deepq import MancalaAgentModel


model_fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join("save", "policy")
world_square_len = int(sys.argv[2]) if len(sys.argv) > 2 else 6

MODEL_SAVE_DIR = 'save'

if torch.cuda.is_available():

    # GPU Config

    device = torch.device('cuda')

else:
    # CPU Config

    device = torch.device('cpu')


os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (30, 100)
os.environ['SDL_VIDEO_CENTERED'] = '0'

world_shape = (world_square_len, world_square_len)

# Load the trained policy net

policy_net = torch.load(os.path.join(os.getcwd(), model_fn), map_location='cpu')
policy_net.eval()

env = MancalaEnv(has_screen=True)
model = MancalaAgentModel(world_shape)

state = env.reset()

done = False

reward_earned = 0
last_steps = []

while not done:

    pos = env.player.position[1] * world_square_len + env.player.position[0]

    env.render()

    input_t = torch.DoubleTensor(state).unsqueeze(0).unsqueeze(0).to(device)

    values = policy_net(input_t).to(device)

    action = torch.argmax(values)

    print("Q values: UP = {}, DOWN = {}, LEFT = {}, RIGHT = {}".format(
        values[0][0],
        values[0][1],
        values[0][2],
        values[0][3]
    ))

    if (pos, action) in last_steps:
        print("Loopy loop")

        actions = [i for i in range(4) if i != action]

        print("New actions: {}".format(actions))

        action = random.sample(actions, 1)[0]

    print("Executing action {}".format(action))

    state, reward, done, info = env.step(action)

    if reward > 0:
        reward_earned += reward
        print("Yay! Got reward: {}, total earned: {}".format(reward, reward_earned))
        last_steps = []
    else:
        last_steps.append((pos, action))

    time.sleep(0.1)

env.close()

print("You died!")

