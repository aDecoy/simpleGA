
#Shows that CPU is better for my work sicne it is a lot faster on smaller networks with small batch size.


import gym
from gym import logger as gym_logger
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import os
import numpy as np
from torch.autograd import Variable
from time import time,gmtime
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class NN_model(nn.Module):
    def __init__(self, num_classes=10):
        super(NN_model, self).__init__()
        self.nettverk = nn.Sequential(
            nn.Linear(5, 3),
            nn.Tanh(),
            nn.Linear(3, 4),
            nn.Tanh(),
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, num_classes), )

    def forward(self, x):
        x = self.nettverk(x)
        return x


use_gpu = False
model = NN_model()

if torch.cuda.device_count() > 1:
    print("USE", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    cudnn.benchmark = True
elif torch.cuda.is_available() and use_gpu :
    model.cuda()
    # print(torch.__version__)
    # print('Using 1 GPU: {}'.format(torch.cuda.get_device_name(0)))
else:
    print("USE ONLY CPU!")

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time()

if use_gpu:
    input = Variable(torch.from_numpy(np.array([1.34, 2.3, 3.4, 5.7, 4.7, 7.7, 2.34, 1.7, 2.7, 3.4, 14.4, 2.4, 3.4, 5.4, 4.4, 7.4, 2.7, 1.3, 2.3, 3.3 ])).float().cuda())
    output = model.forward(input)
    start_time = time()

    for i in range(1000):
        input = Variable(torch.rand(20).float().cuda())
        output = model.forward(input)
else:
    input = Variable(torch.from_numpy(np.array(
        [1.34, 2.3, 3.4, 5.7, 4.7])).float())
    output = model.forward(input)
    start_time = time()

    for i in range(1000):
        input = Variable(torch.rand(5).float())
        output = model.forward(input)

run_time = time() - start_time
print('run_time: {}'.format(run_time))
print(model.parameters)
print(model.nettverk.state_dict())
state_dict1=model.nettverk.state_dict()
print('--------------------')
print(state_dict1.keys())
print(state_dict1['0.weight'][0][0])
state_dict1['0.weight'][0][0]=1.2
print(state_dict1['0.weight'][0][0])
model.nettverk.load_state_dict(state_dict1)
print(model.nettverk.state_dict()['0.weight'][0][0])
print(type(model.nettverk.state_dict()['0.weight'][0]))
print(len(model.nettverk.state_dict()['0.weight'][0]))
# print(model.nettverk.state_dict()['0.weight'][0].item())

print(model.nettverk.state_dict()['0.bias'][0][0])
print(type( model.nettverk.state_dict()['0.bias'][0][0]))
print(len( model.nettverk.state_dict()['0.bias'][0][0]))
print(model.nettverk.state_dict()['0.bias'][0][0].item())



def load_new_model_state(model,state_dict):
    model.nettverk.load_state_dict(state_dict)
    return model



def save_checkpoint(model, generation, checkpoint_file):
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