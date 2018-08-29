import gym
import os
import torch
import numpy as np
from torch.autograd import Variable
from pytorch_model import NN_model


#this must be changed if it is to run in pararell
model= NN_model()
env = gym.make("Ant-v2")
# env = gym.make("Humanoid-v2")
print(env.action_space.high)
print(env.action_space.low)
print(env.observation_space.high)
print(env.observation_space.low)
# env = gym.make("Humanoid-v2")

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
        # print(observation_batch[0])
        # print('observation_batch')
        # print(observation_batch)
        # observation_batch=observation_batch[0]
        prediction = model.forward(Variable(observation_batch))
        # print(prediction)
        action = prediction.data.numpy() #note. dont to argmax(). we are looking for force on all the join, not just one
        # print(action)
        # print('------------------------------------------')
        observations, reward, done, _ = env.step(action)

        total_reward += reward

    observations = env.reset()
    return total_reward



def load_new_model_state(state_dict):
    model.nettverk.load_state_dict(state_dict)
    return model



def save_checkpoint( generation, checkpoint_file):
    torch.save({
        'generation': generation + 1,
        'state_dict': model.state_dict(),
    }, checkpoint_file)
    print('saved checkpoint')

def resume_from_checkpoint(resume_file):
    resume = False
    if resume:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))

            #run_training(0.1, start_epoch)

        else:
            print("=> no checkpoint found at '{}'".format(resume_file))

    else:
        #run_training(learning_rate)
        pass