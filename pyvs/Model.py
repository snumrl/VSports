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

class CombinedSimulationNN(nn.Module):
	def __init__(self,num_states,num_actions):
		super(CombinedSimulationNN, self).__init__()

		# image 40 * 40
		conv1 = nn.Conv2d(4, 4, 5, 2)
		# 18 * 18

		conv2 = nn.Conv2d(4, 6, 3, 1)
		# 16 * 16

		pool1 = nn.MaxPool2d(2)
		# 8 * 8

		conv3 = nn.Conv2d(6, 8, 5, 1)
		# 4 * 4

		self.conv_module = nn.Sequential(
			conv1,
			nn.ReLU(),
			conv2,
			nn.ReLU(),
			pool1,
			conv3,
			nn.ReLU()
		)

		fc1 = nn.Linear(8*4*4, 120)
		fc2 = nn.Linear(120, 84)
		fc3 = nn.Linear(84, 20)

		self.fc_module = nn.Sequential(
			fc1,
			nn.ReLU(),
			fc2,
			nn.ReLU(),
			fc3
		)


		# print(1111)
		# print(use_cuda)

		# if use_cuda:
		# 	self.conv_module = self.conv_module.cuda()
		# 	self.fc_module = self.fc_module.cuda()

		# print(2222)

		num_policyInput = 30

		num_h1 = 256
		num_h2 = 256
		num_h3 = 256


		self.policy = nn.Sequential(
			nn.Linear(num_policyInput, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_h3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h3, num_actions)
		)
		self.value = nn.Sequential(
			nn.Linear(num_policyInput, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_h3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h3,1)
		)

		# if use_cuda:
		# 	self.policy = self.policy.cuda()
		# 	self.value = self.value.cuda()
		self.log_std = nn.Parameter(torch.zeros(num_actions))
		self.conv_module.apply(weights_init)
		self.fc_module.apply(weights_init)
		self.policy.apply(weights_init)
		self.value.apply(weights_init)

	def forward(self,x):
		# print(x.size())
		# exit(0)
		x = x.cuda()
		mapRow = 40
		mapCol = 40
		numLayer = 4

		# mapX = Tensor([x[0][:numLayer*mapCol*mapRow].cpu().detach().numpy()])
		# mapX = mapX.view(numLayer, mapCol, mapRow);

		# vecX = Tensor([x[0][numLayer*mapCol*mapRow:].cpu().detach().numpy()])

		# mapX = torch.split(x, numLayer*mapCol*mapRow, dim=1)
		# mapX = Tensor([]*1);
		mapX = x[0][10:numLayer*mapCol*mapRow+10].view(1, numLayer, mapCol, mapRow)
		vecX = x[0][:10].view(1, -1)
		# print(mapX.size())
		# print(vecX.size())
		# print(mapX)
		# start = time.process_time()
		mapX_ = self.conv_module(mapX)
		# print(time.process_time()-start)
		# print(mapX_.size())
		# print(mapX_.size())
		# start = time.process_time()

		# vecX = torch.cat()
		dim = 1
		# out.size() -> Torch.size([batch_size, layers, width, height])
		for d in mapX_.size()[1:]:
			dim = dim * d
		mapX_ = mapX_.view(-1, dim)
		mapX_ = self.fc_module(mapX_)
		# exit()
		# return out
		# print(time.process_time()-start)

		concatVecX = torch.cat((vecX[0],mapX_[0]),0).view(1,-1)

		# print(concatVecX.size())
		# exit()
		# return MultiVariateNormal(self.policy(concatVecX),self.log_std.exp()),self.value(concatVecX);
		return MultiVariateNormal(self.policy(concatVecX),self.log_std.exp()),self.value(concatVecX);

	def load(self,path):
		print('load simulation nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self,path):
		print('save simulation nn {}'.format(path))
		torch.save(self.state_dict(),path)

	def get_action(self,s):
		ts = torch.tensor(s)
		# print(ts.size())
		p,_ = self.forward(ts.view(1,-1))
		# print(p.size())
		return p.loc.cpu().detach().numpy()

	def get_random_action(self,s):
		ts = torch.tensor(s)
		p,_ = self.forward(ts.view(1,-1))
		return p.sample().cpu().detach().numpy()

# class StateCNN(nn.Module):
# 	def __init__(self):
# 		super(StateCNN, self).__init__()
# 		#image 40 * 40
# 		conv1 = nn.Conv2d(4, 6, 5, 2)
# 		# 18 * 18

# 		conv2 = nn.Conv2d(6, 10, 3, 1)
# 		# 16 * 16

# 		pool1 = nn.MaxPool2d(2)
# 		# 8 * 8

# 		conv3 = nn.Conv2d(10, 16, 5, 1)
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


# class StateLSTM(nn.Module):
# 	def __init__(self):
# 		super(StateLSTM, self).__init__()