import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn

#Ant v2
class NN_model(nn.Module):
    def __init__(self, num_classes=8):
        super(NN_model, self).__init__()
        self.nettverk = nn.Sequential(
            nn.Linear(111, 200),
            nn.Tanh(),
            nn.Linear(200, 300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.Tanh(),
            nn.Linear(200, num_classes), )

    def forward(self, x):
        x = self.nettverk(x)
        return x
#         #Humanoid v2
# class NN_model(nn.Module):
#     def __init__(self, num_classes=17):
#         super(NN_model, self).__init__()
#         self.nettverk = nn.Sequential(
#             nn.Linear(376, 200),
#             nn.Tanh(),
#             nn.Linear(200, 300),
#             nn.Tanh(),
#             nn.Linear(300, 200),
#             nn.Tanh(),
#             nn.Linear(200, num_classes), )
#
#     def forward(self, x):
#         x = self.nettverk(x)
#         return x

def get_random_state_dict():
    return NN_model().nettverk.state_dict()