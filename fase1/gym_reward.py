import gym
import copy
import torch
import numpy as np
from torch.autograd import Variable
from pytorch_model import NN_model


#this must be changed if it is to run in pararell
model= NN_model()
env = gym.make("Humanoid-v2")

def get_reward(state_dict, render=False):
    # cloned_model = copy.deepcopy(model)

    # cloned_model.nettverk.load_state_dict(state_dict)
    model.nettverk.load_state_dict(state_dict)


    # env._max_episode_steps=500
    observations = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            # time.sleep(0.05)
        observation_batch = torch.from_numpy(observations[np.newaxis,...]).float()
        prediction = model(Variable(observation_batch))
        action = prediction.data.numpy() #note. dont to argmax(). we are looking for force on all the join, not just one
        observations, reward, done, _ = env.step(action)

        total_reward += reward

    observations = env.reset()
    return total_reward


