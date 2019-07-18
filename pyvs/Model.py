import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
import time
from collections import OrderedDict
import numpy as np
from IPython import embed

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self,val).sum(-1, keepdim = True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def weights_init(m):
	classname =	m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight)
		m.bias.data.zero_()

class SimulationNN(nn.Module):
	def __init__(self,num_states,num_actions):
		super(SimulationNN, self).__init__()

		num_h1 = 256
		num_h2 = 256
		num_h3 = 256

		self.policy = nn.Sequential(
			nn.Linear(num_states, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_h3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h3, num_actions)
		)
		self.value = nn.Sequential(
			nn.Linear(num_states, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_h3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h3,1)
		)
		self.log_std = nn.Parameter(torch.zeros(num_actions))
		self.policy.apply(weights_init)
		self.value.apply(weights_init)

	def forward(self,x):
		return MultiVariateNormal(self.policy(x),self.log_std.exp()),self.value(x);

	def load(self,path):
		print('load simulation nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self,path):
		print('save simulation nn {}'.format(path))
		torch.save(self.state_dict(),path)

	def get_action(self,s):
		ts = torch.tensor(s)
		p,_ = self.forward(ts)
		return p.loc.cpu().detach().numpy()

	def get_random_action(self,s):
		ts = torch.tensor(s)
		p,_ = self.forward(ts)
		return p.sample().cpu().detach().numpy()


# class CNN(nn.Moudle):
# 	def __init__(self):
# 		super(CNNClassifier, self).__init__()
# 		#image 80 * 80
# 		conv1 = nn.Conv2d(1, 6, 9, 4)
# 		# 18 * 18

# 		conv2 = nn.Conv2d(1, 16, 3, 1)
# 		# 16 * 16

# 		pool1 = nn.MaxPool2d(2)
# 		# 8 * 8

# 		conv3 = nn.Conv2d(1, 16, 5, 1)
# 		# 4 * 4

# 		self.conv_module = nn.Sequential(
# 			conv1,
# 			nn.ReLU(),
# 			conv2,
# 			nn.ReLU(),
# 			pool1,
# 			conv3,
# 			nn.ReLU()
# 		)

# 		fc1 = nn.Linear(16*4*4, 120)

# 		fc2 = nn.Linear(120, 84)

# 		fc3 = nn.Linear(84, 20)

# 		self.fc_module = nn.Sequential(
# 			fc1,
# 			nn.ReLU(),
# 			fc2,
# 			nn.ReLU(),
# 			fc3
# 		)

# 		if use_cuda:
# 			self.conv_moudle = self.conv_module.cuda()
# 			self.fc_module = self.fc_module.cuda()


# 	def forward(self, x):
# 		out = self.conv_module(x)

# 		dim = 1
# 		# out.size() -> Torch.size([batch_size, layers, width, height])
# 		for d in out.size()[1:]:
# 			dim = dim * d

# 		out = out.view(-1, dim)

# 		out = self.fc_module(out)
# 		return out


# class LSTM(nn.Module):
# 	def __init__(self):
# 		super(LSTM, self).__init__()