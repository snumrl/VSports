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
		# exit()
		# image 40 * 40
		conv1 = nn.Conv2d(5, 5, 5, 2)
		# 18 * 18

		conv2 = nn.Conv2d(5, 8, 3, 1)
		# 16 * 16

		pool1 = nn.MaxPool2d(2)
		# 8 * 8

		conv3 = nn.Conv2d(8, 8, 5, 1)
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

		self.hidden_size = 60
		self.num_layers = 3

		# self.prev_hidden = self.init_hidden(1)
		# self.prev_cell = self.init_hidden(1)

		# self.rnn = nn.LSTM(num_policyInput, self.hidden_size, num_layers=self.num_layers,bias=True,batch_first=True,bidirectional=True)
		
		self.rnn = nn.LSTM(num_policyInput, self.hidden_size, num_layers=self.num_layers)#,bias=True,batch_first=True,bidirectional=True)




		num_h1 = 128
		num_h2 = 128
		num_h3 = 128


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


		# x = x.cuda()
		mapRow = 40
		mapCol = 40
		numLayer = 5

		x = x.cuda()
		mapX = x.narrow(1, 10, numLayer*mapCol*mapRow).view(-1, numLayer, mapCol, mapRow)

		# .view(1, numLayer, mapCol, mapRow)

		# print(x.size(), end=' ')
		# vecX = x.view(1, 10)
		
		# vecX = x[0][:10].view(1, -1)
		vecX = x.narrow(1,0,10)
		
		# print(vecX.size())
		# exit()
		# print(vecX.requires_grad)
		# exit()
		mapX_ = self.conv_module(mapX)

		dim = 1
		for d in mapX_.size()[1:]:
			dim = dim * d
		mapX_ = mapX_.view(-1, dim)
		mapX_ = self.fc_module(mapX_)


		# print(vecX.size(), end=" ")
		# print(mapX_.size())
		concatVecX = torch.cat((vecX,mapX_),1)
		# print(concatVecX.size(), end = " ")
		concatVecX = concatVecX.view(-1, 30)


		batch_size = x.size(0)

		# Initializing hidden state for first input using method defined below
		hidden = self.init_hidden(batch_size)
		cell = self.init_hidden(batch_size)


		# concatVecX = torch.cat((mapX_, concatVecX), 1)

		# we will see infinite timesteps. Hmm.. or just 5 timestep?
		# concatVecX = concatVecX.view(1, batch_size, -1)
		# print(concatVecX.size())
		# rnnOutput, (hidden, cell) = self.rnn(concatVecX, (hidden, cell) )
		# print(hidden.size())


		# rnnOutput, (a,b) = self.rnn(concatVecX)

		# print(a.size())

		# print(rnnOutput.size())
		# exit()

		# print(concatVecX.size())
		# concatVecX = torch.cat((vecX[0],mapX_[0]),0).view(1,-1)

		return MultiVariateNormal(self.policy(concatVecX),self.log_std.exp()),self.value(concatVecX);
		# return MultiVariateNormal(self.policy(vecX),self.log_std.exp()),self.value(vecX);
		# return MultiVariateNormal(self.policy(x),self.log_std.exp()),self.value(x);


	def forward_rnn(self,x, hidden, cell):
		# print(x.size())
		# exit(0)


		# x = x.cuda()
		mapRow = 40
		mapCol = 40
		numLayer = 5

		x = x.cuda()
		mapX = x.narrow(1, 10, numLayer*mapCol*mapRow).view(-1, numLayer, mapCol, mapRow)

		# .view(1, numLayer, mapCol, mapRow)

		# print(x.size(), end=' ')
		# vecX = x.view(1, 10)
		
		# vecX = x[0][:10].view(1, -1)
		vecX = x.narrow(1,0,10)
		
		# print(vecX.size())
		# exit()
		# print(vecX.requires_grad)
		# exit()
		mapX_ = self.conv_module(mapX)

		dim = 1
		for d in mapX_.size()[1:]:
			dim = dim * d
		mapX_ = mapX_.view(-1, dim)
		mapX_ = self.fc_module(mapX_)


		# print(vecX.size(), end=" ")
		# print(mapX_.size())
		concatVecX = torch.cat((vecX,mapX_),1)
		# print(concatVecX.size(), end = " ")
		concatVecX = concatVecX.view(-1, 30)


		batch_size = x.size(0)

		# Initializing hidden state for first input using method defined below
		if hidden == None :
			hidden = self.init_hidden(batch_size)

		if cell == None :
			cell = self.init_hidden(batch_size)


		# concatVecX = torch.cat((mapX_, concatVecX), 1)

		# we will see infinite timesteps. Hmm.. or just 5 timestep?
		# concatVecX = concatVecX.view(1, batch_size, -1)
		# print(concatVecX.size())
		# rnnOutput, (hidden, cell) = self.rnn(concatVecX, (hidden, cell) )
		# print(hidden.size())


		rnnOutput, (a,b) = self.rnn(concatVecX, (hidden, cell))

		# print(a.size())

		# print(rnnOutput.size())
		# exit()

		# print(concatVecX.size())
		# concatVecX = torch.cat((vecX[0],mapX_[0]),0).view(1,-1)

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
	
	def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
		hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
		return hidden

class NoCNNSimulationNN(nn.Module):
	def __init__(self,num_states,num_actions):
		super(NoCNNSimulationNN, self).__init__()
		# exit()
		# image 40 * 40
		conv1 = nn.Conv2d(5, 5, 5, 2)
		# 18 * 18

		conv2 = nn.Conv2d(5, 8, 3, 1)
		# 16 * 16

		pool1 = nn.MaxPool2d(2)
		# 8 * 8

		conv3 = nn.Conv2d(8, 8, 5, 1)
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
		self.useMap = False
		self.num_policyInput = 0
		if self.useMap:
			self.num_policyInput = 22
		else:
			self.num_policyInput = 26

		self.hidden_size = 256
		self.num_layers = 1

		# self.prev_hidden = self.init_hidden(1)
		# self.prev_cell = self.init_hidden(1)

		# self.rnn = nn.LSTM(num_policyInput, self.hidden_size, num_layers=self.num_layers,bias=True,batch_first=True,bidirectional=True)
		
		self.rnn = nn.LSTM(self.num_policyInput, self.hidden_size, num_layers=self.num_layers)#,bias=True,batch_first=True,bidirectional=True)


		self.cur_hidden = self.init_hidden(1)


		num_h1 = 128
		num_h2 = 128
		num_h3 = 128


		self.policy = nn.Sequential(
			nn.Linear(self.num_policyInput, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_h3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h3, num_actions)
		)
		self.value = nn.Sequential(
			nn.Linear(self.num_policyInput, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_h3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h3,1)
		)

		self.policy_rnn = nn.Sequential(
			nn.Linear(self.hidden_size, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_h3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h3, num_actions)
		)
		self.value_rnn = nn.Sequential(
			nn.Linear(self.hidden_size, num_h1),
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
		self.policy_rnn.apply(weights_init)
		self.value_rnn.apply(weights_init)

	def forward(self,x):
		concatVecX = x.cuda()
		return MultiVariateNormal(self.policy(concatVecX),self.log_std.exp()),self.value(concatVecX);

	def forward_rnn(self,x, in_hidden):
		# print(x.size())
		# exit(0)

		# self.useMap = False

		# x = x.squeeze()
		numNumberState = 2

		if self.useMap is True:
			mapRow = 40
			mapCol = 40
			numLayer = 5

			x = x.cuda()
			mapX = x.narrow(1, numNumberState, numLayer*mapCol*mapRow).view(-1, numLayer, mapCol, mapRow)

			vecX = x.narrow(1,0,numNumberState)
			

			mapX_ = self.conv_module(mapX)

			dim = 1
			for d in mapX_.size()[1:]:
				dim = dim * d
			mapX_ = mapX_.view(-1, dim)
			mapX_ = self.fc_module(mapX_)

			concatVecX = torch.cat((vecX,mapX_),1)
			concatVecX = concatVecX.view(-1, self.num_policyInput)
		else :
			concatVecX = x.cuda()

		# Initializing hidden state for first input using method defined below
		batch_size = concatVecX.size()[0];

		useRNN = False

		if useRNN :
			rnnOutput, out_hidden = self.rnn(concatVecX.view(1, batch_size,-1), in_hidden)
			return MultiVariateNormal(self.policy_rnn(rnnOutput),self.log_std.exp()),self.value_rnn(rnnOutput), out_hidden
		else :
			return MultiVariateNormal(self.policy(concatVecX).unsqueeze(0),self.log_std.exp()),self.value(concatVecX).unsqueeze(0), in_hidden


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

	def get_action_rnn(self,s):
		ts = torch.tensor(s)
		# print(ts.size())
		# if in_hidden is None:
		# 	new_hidden = self.init_hidden
		# else :
		# 	new_hidden = in_hidden
		p,_1 ,new_hidden= self.forward_rnn(ts.unsqueeze(0), self.cur_hidden)
		# new_hidden = self.cur_hidden
		# new_hidden = self.cur_hidden
		self.cur_hidden = new_hidden
		# print(p.size())
		return p.loc.cpu().detach().numpy()

	def reset_hidden(self):
		self.cur_hidden = self.init_hidden(1)

	# def get_random_action(self,s):
	# 	ts = torch.tensor(s)
	# 	p,_ = self.forward(ts.view(1,-1))
	# 	return p.sample().cpu().detach().numpy()
	
	def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
		hidden = (Tensor(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()),Tensor(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()))
		return hidden