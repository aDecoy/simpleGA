import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from numpy import linalg as LA
#Ant v2
# class NN_model(nn.Module):
#     def __init__(self, num_classes=8):
#         super(NN_model, self).__init__()
#         self.nettverk = nn.Sequential(
#             nn.Linear(111, 200),
#             nn.Tanh(),
#             nn.Linear(200, 150),
#             nn.Tanh(),
#             nn.Linear(150, 70),
#             nn.Tanh(),
#             nn.Linear(70, num_classes), )
#
#     def forward(self, x):
#         x = self.nettverk(x)
#         return x
#         #Humanoid v2
# class NN_model(nn.Module):
#     def __init__(self, num_classes=17):
#         super(NN_model, self).__init__()
#         self.linear1 = nn.Linear(376, 100)
#         self.linear2 =nn.Linear(100, 50)
#         self.linear3 =nn.Linear(50, num_classes)
#
#         # self.m = nn.SELU()
#         self.hard_tanh = nn.Hardtanh(min_val=-.4, max_val=0.4)
#         self.tanh = nn.Tanh()
#         self.shrink = nn.Hardshrink()
#         self.norm = nn.LayerNorm(((1,376)))
#
#         # self.nettverk = nn.Sequential(
#         #     nn.Linear(376, 200),
#         #     # nn.Tanh(),
#         #     nn.Linear(200, 300),
#         #     # nn.Tanh(),
#         #     nn.Linear(300, 200),
#         #     # nn.Tanh(),
#         #     nn.Linear(200, num_classes), )
#
#     def forward(self, x):
#         # x = self.nettverk(x)
#         # print('layer input ------------------------------------------------')
#         # print(x)
#         # x = self.norm(x)
#         x= self.linear1(x)
#         # print('layer 1 ------------------------------------------------')
#         # print(x)
#         x= self.linear2(x)
#         x= self.tanh(x)
#         # print('layer 2 ------------------------------------------------')
#         # print(x)
#         x= self.linear3(x)
#         x= self.tanh(x)
#         # print('layer 3 ------------------------------------------------')
#         # print(x)
#
#         return x
# LunarLanderContinuous-v2
class NN_model(nn.Module):
    def __init__(self, num_classes=17):
        super(NN_model, self).__init__()
        self.linear1 = nn.Linear(376, 100)
        self.linear2 =nn.Linear(100, 50)
        self.linear3 =nn.Linear(50, num_classes)

        # self.m = nn.SELU()
        self.hard_tanh = nn.Hardtanh(min_val=-.4, max_val=0.4)
        self.tanh = nn.Tanh()
        self.shrink = nn.Hardshrink()
        self.norm = nn.LayerNorm(((1,376)))

    def forward(self, x):
        # x = self.nettverk(x)
        # print('layer input ------------------------------------------------')
        # print(x)
        # x = self.norm(x)
        x= self.linear1(x)
        # print('layer 1 ------------------------------------------------')
        # print(x)
        x= self.linear2(x)
        x= self.tanh(x)
        # print('layer 2 ------------------------------------------------')
        # print(x)
        x= self.linear3(x)
        x= self.tanh(x)
        # print('layer 3 ------------------------------------------------')
        # print(x)

        return x

def get_random_state_dict():
    return NN_model().state_dict()

