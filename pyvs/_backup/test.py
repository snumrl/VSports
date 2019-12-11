import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from Model import *
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# Tensor = FloatTensor

num_state = 6410
num_action = 3

model = CombinedSimulationNN(num_state, num_action).cuda()
state = [0.0]*num_state
action = model.get_action(state)
print(action)