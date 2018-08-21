import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn

import os
import numpy as np
from torch.autograd import Variable
from time import time,gmtime
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class NN_model(nn.Module):
    def __init__(self, num_classes=10):
        super(NN_model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(20, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 2000),
            nn.LeakyReLU(),

            nn.Linear(2000, 200),
            nn.LeakyReLU(),
            nn.Linear(200, num_classes), )

    def forward(self, x):
        x = self.classifier(x)
        return x


use_gpu = False
model = NN_model()

if torch.cuda.device_count() > 1:
    print("USE", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    cudnn.benchmark = True
elif torch.cuda.is_available() and use_gpu :
    model.cuda()
    print(torch.__version__)
    print('Using 1 GPU: {}'.format(torch.cuda.get_device_name(0)))
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
        [1.34, 2.3, 3.4, 5.7, 4.7, 7.7, 2.34, 1.7, 2.7, 3.4, 14.4, 2.4, 3.4, 5.4, 4.4, 7.4, 2.7, 1.3, 2.3,
         3.3])).float())
    output = model.forward(input)
    start_time = time()

    for i in range(1000):
        input = Variable(torch.rand(20).float())
        output = model.forward(input)

run_time = time() - start_time
print('run_time: {}'.format(run_time))
