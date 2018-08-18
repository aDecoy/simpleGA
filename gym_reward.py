import gym
import copy
import torch
import numpy as np
from torch.autograd import Variable


def get_reward(model,state_dict, render=False):
    cloned_model = copy.deepcopy(model)

    cloned_model.nettverk.load_state_dict(state_dict)

    env = gym.make("Humanoid-v2")
    # env._max_episode_steps=500
    observations = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            # time.sleep(0.05)
        observation_batch = torch.from_numpy(observations[np.newaxis,...]).float()
        prediction = cloned_model(Variable(observation_batch))
        action = prediction.data.numpy() #note. dont to argmax(). we are looking for force on all the join, not just one
        observations, reward, done, _ = env.step(action)

        total_reward += reward

    env.close()
    return total_reward