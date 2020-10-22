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
import copy
from Utils import RunningMeanStd

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
	def __init__(self, num_states, num_actions):
		super(SimulationNN, self).__init__()

		self.num_policyInput = num_states

		num_h1 = 128
		num_h2 = 128
		# num_h3 = 256

		self.policy = nn.Sequential(
			nn.Linear(num_states, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_actions),
			# nn.Tanh()
			# nn.LeakyReLU(0.2, inplace=True),
			# nn.Linear(num_h3, num_actions)
		)

		self.log_std = nn.Parameter(-1.0 * torch.ones(num_actions))

		self.policy.apply(weights_init)

	def forward(self,x):
		x = x.cuda()

		batch_size = x.size()[0];

		# return MultiVariateNormal(self.ss_policy(x).unsqueeze(0),self.log_std.exp())
		return self.policy(x)
	def load(self,path):
		print('load policy nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self,path):
		print('save imitation policy nn {}'.format(path))
		torch.save(self.state_dict(),path)

	def get_action(self,s):
		ts = torch.tensor(s)

		p = self.forward(ts.unsqueeze(0))

		# return p.loc.cpu().detach().numpy()
		return p.cpu().detach().numpy()



class ActorCriticNN(nn.Module):
	def __init__(self, num_states, num_actions, log_std = 0.0, softmax = False, actionType = False):
		super(ActorCriticNN, self).__init__()
		self.softmax = softmax
		self.num_policyInput = num_states

		self.hidden_size = 128
		self.num_layers = 1
		self.actionType = actionType
		self.softmax = softmax

		# self.rnn = nn.LSTM(self.num_policyInput, self.hidden_size, num_layers=self.num_layers)
		# self.cur_hidden = self.init_hidden(1)

		num_h1 = 256
		num_h2 = 256
		# num_h3 = 256
		# self.policy = None

		if self.softmax :
			self.policy = nn.Sequential(
				nn.Linear(self.num_policyInput, num_h1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(num_h1, num_h2),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(num_h2, num_actions),
				# nn.Softmax(dim = 1)
				# nn.Tanh()
				# nn.LeakyReLU(0.2, inplace=True),
				# nn.Linear(num_h3, num_actions)
			)
		else:
			self.policy = nn.Sequential(
				nn.Linear(self.num_policyInput, num_h1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(num_h1, num_h2),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(num_h2, num_actions),
				# nn.Tanh()
				# nn.LeakyReLU(0.2, inplace=True),
				# nn.Linear(num_h3, num_actions)
			)
		self.value = nn.Sequential(
			nn.Linear(self.num_policyInput, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, 1),
			# nn.LeakyReLU(0.2, inplace=True),
			# nn.Linear(num_h3, 1)
		)


		self.log_std = nn.Parameter(log_std * torch.ones(num_actions))
		# self.log_std = nn.Parameter(Tensor([0, 0, -2]))

		# self.rnn.apply(weights_init)
		self.policy.apply(weights_init)

		self.rms = RunningMeanStd(shape=(num_states-2))

	def loadRMS(self, path):
		print('load RMS : {}'.format(path))
		self.rms.load(path)


	def forward(self,x):
		# self.rms.apply(x)
		x = x.cuda()

		# batch_size = x.size()[0];

		action = self.policy(x)
	
		if self.actionType:
			mask = x[:,-2:]
			action = torch.mul(action, mask)


			
			# embed()
			# exit(0)


		# embed()
		# exit(0)
		if self.softmax:
			sm =  nn.Softmax(dim = 1)
			action = sm(action)

		# self.log_std = nn.Parameter(Tensor([0, 0, -2]))
		# k = np.exp(-0.01*num_eval)
		# rnnOutput, out_hidden = self.rnn(x.view(1, batch_size,-1), in_hidden)
		return MultiVariateNormal(action.unsqueeze(0),self.log_std.exp()), self.value(x)
		# return MultiVariateNormal(self.policy(rnnOutput).unsqueeze(0),self.log_std.exp()), self.value(rnnOutput), out_hidden

	# def forwardAndUpdate(self,x):



	# 	x = x.cuda()

	# 	batch_size = x.size()[0];

	# 	# self.log_std = nn.Parameter(Tensor([0, 0, -2]))
	# 	# k = np.exp(-0.01*num_eval)
	# 	# rnnOutput, out_hidden = self.rnn(x.view(1, batch_size,-1), in_hidden)
	# 	return MultiVariateNormal(self.policy(x).unsqueeze(0),self.log_std.exp()), self.value(x)
	# 	# return MultiVariateNormal(self.policy(rnnOutput).unsqueeze(0),self.log_std.exp()), self.value(rnnOutput), out_hidden


	def load(self,path):
		print('load nn {}'.format(path))
		# embed()
		# exit(0)	
		self.load_state_dict(torch.load(path))

	def save(self,path):
		print('save nn {}'.format(path))
		torch.save(self.state_dict(),path)

	def get_action(self,s):
		# embed()
		# exit(0)
		s[0:len(self.rms.mean)] = self.rms.applyOnly(s[0:len(self.rms.mean)])
		ts = torch.tensor(s)

		# embed()
		# exit(0)
		p, _v= self.forward(ts.unsqueeze(0))

		# self.cur_hidden = new_hidden
		# print(p.loc.cpu().detach().numpy())
		# return p.sample().cpu().detach().numpy()
		return p.sample().cpu().detach().numpy().astype(np.float32)
		# return p.loc.cpu().detach().numpy().astype(np.float32)

	def get_value(self, s):
		ts = torch.tensor(s)

		_p, v= self.forward(ts.unsqueeze(0))

		# self.cur_hidden = new_hidden

		# return p.sample().cpu().detach().numpy()
		return v.cpu().detach().numpy()[0]

	def get_actionNoise(self,s):
		ts = torch.tensor(s)

		p, _v= self.forward(ts.unsqueeze(0))

		# self.cur_hidden = new_hidden
		# print(p.loc.cpu().detach().numpy())

		# return p.sample().cpu().detach().numpy()
		return p.sample.cpu().detach().numpy() - p.loc.cpu().detach().numpy()




class ActorNN(nn.Module):
	def __init__(self, num_states, num_actions, log_std = 0.0):
		super(ActorNN, self).__init__()
		self.num_policyInput = num_states

		self.hidden_size = 128
		self.num_layers = 1


		num_h1 = 256
		num_h2 = 256

		self.policy = nn.Sequential(
			nn.Linear(self.num_policyInput, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_actions),
		)

		self.log_std = nn.Parameter(log_std * torch.ones(num_actions))


		self.policy.apply(weights_init)

		self.rms = RunningMeanStd(shape=(num_states))

	def loadRMS(self, path):
		print('load RMS : {}'.format(path))
		self.rms.load(path)


	def forward(self,x):
		x = x.cuda()

		batch_size = x.size()[0];

		return MultiVariateNormal(self.policy(x).unsqueeze(0),self.log_std.exp())



	def load(self,path):
		print('load nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self,path):
		print('save nn {}'.format(path))
		torch.save(self.state_dict(),path)

	def get_action(self,s):
		# embed()
		# exit(0)
		# s[0:len(self.rms.mean)] = self.rms.applyOnly(s[0:len(self.rms.mean)])
		ts = torch.tensor(s)

		# embed()
		# exit(0)
		p = self.forward(ts.unsqueeze(0))


		return p.sample().cpu().detach().numpy().astype(np.float32)
		# return p.loc.cpu().detach().numpy().astype(np.float32)




class CriticNN(nn.Module):
	def __init__(self, num_states):
		super(CriticNN, self).__init__()
		self.num_policyInput = num_states

		self.hidden_size = 128
		self.num_layers = 1

		# self.rnn = nn.LSTM(self.num_policyInput, self.hidden_size, num_layers=self.num_layers)
		# self.cur_hidden = self.init_hidden(1)

		num_h1 = 256
		num_h2 = 256
		# num_h3 = 256
		# self.policy = None

		self.value = nn.Sequential(
			nn.Linear(self.num_policyInput, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, 1),
			# nn.LeakyReLU(0.2, inplace=True),
			# nn.Linear(num_h3, 1)
		)

		self.value.apply(weights_init)

		self.rms = RunningMeanStd(shape=(num_states))

	def loadRMS(self, path):
		print('load RMS : {}'.format(path))
		self.rms.load(path)


	def forward(self,x):
		x = x.cuda()

		return self.value(x)



	def load(self,path):
		print('load nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self,path):
		print('save nn {}'.format(path))
		torch.save(self.state_dict(),path)


	def get_value(self, s):
		ts = torch.tensor(s)

		v= self.forward(ts.unsqueeze(0))

		return v.cpu().detach().numpy()[0]

	# def get_value_gradient(self,s):
	# 	delta = 0.01
	# 	value_gradient = [None]*len(s)
	# 	# print(s)
	# 	for i in range(len(s)):
	# 		positive_s = copy.deepcopy(s)
	# 		positive_s[i] += delta
	# 		negative_s = copy.deepcopy(s)
	# 		negative_s[i] += -delta
	# 		# print(positive_s)
	# 		# print(negative_s)
	# 		# print("############")

	# 		# print("1111")
	# 		_p, positive_v, new_hidden = self.forward(torch.tensor(positive_s).unsqueeze(0), self.cur_hidden)
	# 		# print("2222")
	# 		_p, negative_v, new_hidden = self.forward(torch.tensor(negative_s).unsqueeze(0), self.cur_hidden)
	# 		# print(positive_v[0],end=" ")
	# 		# print(negative_v[0])
	# 		# print("3333")
	# 		value_gradient[i] = (positive_v.cpu().detach().numpy()[0]-negative_v.cpu().detach().numpy()[0])/(delta*2.0)
	# 		# print(positive_v-negative_v)
	# 		# print("4444")

	# 	# value_gradient = s
	# 	# print(value_gradient)
	# 	return value_gradient


	# def reset_hidden(self):
	# 	self.cur_hidden = self.init_hidden(1)

 #    # This method generates the first hidden state of zeros which we'll use in the forward pass
	# def init_hidden(self, batch_size):
	# 	hidden = (Tensor(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()),\
	# 			Tensor(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()))
	# 	return hidden



class RandomNN(nn.Module):
	def __init__(self, num_states, num_features):
		super(RandomNN, self).__init__()

		self.num_policyInput = num_states

		num_h1 = 128
		num_h2 = 128

		self.network = nn.Sequential(
			nn.Linear(self.num_policyInput, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_features),
		)

		self.network.apply(weights_init)

	def forward(self,x):
		x = x.cuda()

		batch_size = x.size()[0];

		return self.network(x)

	def load(self,path):
		print('load random network nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self,path):
		print('save random network nn {}'.format(path))
		torch.save(self.state_dict(),path)

	def get_feature(self,s):
		ts = torch.tensor(s)

		p = self.forward(ts.unsqueeze(0))

		# return p.loc.cpu().detach().numpy()
		return p.cpu().detach().numpy()

# class ExplorationNN(nn.Module):
# 	def __init__(self, num_states, num_actions):
# 		super(ExplorationNN, self).__init__()

# 		self.num_policyInput = num_states

# 		self.hidden_size = 128
# 		self.num_layers = 1

# 		# self.rnn = nn.LSTM(self.num_policyInput, self.hidden_size, num_layers=self.num_layers)
# 		# self.cur_hidden = self.init_hidden(1)

# 		num_h1 = 128
# 		num_h2 = 128
# 		# num_h3 = 256

# 		self.policy = nn.Sequential(
# 			nn.Linear(self.num_policyInput, num_h1),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Linear(num_h1, num_h2),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Linear(num_h2, num_actions),
# 			# nn.Tanh()
# 			# nn.LeakyReLU(0.2, inplace=True),
# 			# nn.Linear(num_h3, num_actions)
# 		)
# 		self.value = nn.Sequential(
# 			nn.Linear(self.num_policyInput, num_h1),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Linear(num_h1, num_h2),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Linear(num_h2, 1),
# 			# nn.LeakyReLU(0.2, inplace=True),
# 			# nn.Linear(num_h3, 1)
# 		)


# 		# self.log_std = nn.Parameter(-1.0 * torch.ones(num_actions))
# 		self.log_std = nn.Parameter(Tensor([-1, -1, -2, -2]))

# 		# self.rnn.apply(weights_init)
# 		self.policy.apply(weights_init)
# 		self.value.apply(weights_init)

# 	def forward(self,x):
# 		x = x.cuda()

# 		batch_size = x.size()[0];

# 		# rnnOutput, out_hidden = self.rnn(x.view(1, batch_size,-1), in_hidden)
# 		return MultiVariateNormal(self.policy(x).unsqueeze(0),self.log_std.exp()), self.value(x)
# 		# return MultiVariateNormal(self.policy(rnnOutput).unsqueeze(0),self.log_std.exp()), self.value(rnnOutput), out_hidden
# 	def load(self,path):
# 		# print('load nn {}'.format(path))
# 		self.load_state_dict(torch.load(path))

# 	def save(self,path):
# 		print('save nn {}'.format(path))
# 		torch.save(self.state_dict(),path)

# 	def get_action(self,s):
# 		ts = torch.tensor(s)

# 		p, _v= self.forward(ts.unsqueeze(0))

# 		# self.cur_hidden = new_hidden
# 		# print(p.loc.cpu().detach().numpy())

# 		return p.sample().cpu().detach().numpy()
# 		# return p.loc.cpu().detach().numpy()

# 	def get_value(self, s):
# 		ts = torch.tensor(s)

# 		_p, v= self.forward(ts.unsqueeze(0))

# 		# self.cur_hidden = new_hidden

# 		# return p.sample().cpu().detach().numpy()
# 		return v.cpu().detach().numpy()[0]
