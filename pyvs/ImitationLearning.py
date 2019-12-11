import math
import random
import time
import os
import sys
import copy
from datetime import datetime

import collections
from collections import namedtuple
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from pyvs import Env
from IPython import embed
from Model import *
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


RNNEpisode = namedtuple('RNNEpisode', ('s','a','r','value','logprob','hidden'))

class RNNEpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def push(self, *args):
		self.data.append(RNNEpisode(*args))

	def getData(self):
		return self.data

RNNTransition = namedtuple('RNNTransition',('s','a','logprob','TD','GAE','hidden_h','hidden_c'))

HumanEpisode = namedtuple('HumanEpisode', ('s','a'))
class HumanEpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def push(self, *args):
		self.data.append(HumanEpisode(*args))

	def getData(self):
		return self.data

SimpleTransition = namedtuple('SimpleTransition',('s','a'))



class RNNReplayBuffer(object):
	def __init__(self, buff_size = 10000):
		super(RNNReplayBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def push(self,*args):
		self.buffer.append(RNNTransition(*args))

	def clear(self):
		self.buffer.clear()

class Buffer(object):
	def __init__(self, buff_size = 10000):
		super(Buffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def push(self,replay_buffer):
		self.buffer.append(replay_buffer)

	def clear(self):
		self.buffer.clear()

class ImitationLearning(object):
	def __init__(self):
		np.random.seed(seed = int(time.time()))
		self.num_slaves = 8
		self.num_agents = 4
		self.env = Env(self.num_agents)
		self.num_state = self.env.getNumState()
		self.num_action = self.env.getNumAction()

		self.num_epochs = 10
		self.num_evaluation = 0
		self.num_tuple_so_far = 0
		self.num_episode = 0
		self.num_tuple = 0

		self.num_simulation_Hz = self.env.getSimulationHz()
		self.num_control_Hz = self.env.getControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.lb = 0.95

		self.buffer_size = 8*1024
		self.batch_size = 512

		self.buffer = Buffer(30000)

		self.model = [None]*self.num_slaves*self.num_agents
		for i in range(self.num_slaves*self.num_agents):
			# exit()
			self.model[i] = SimulationNN(self.num_state, self.num_action)

			if use_cuda:
				self.model[i].cuda()

		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio
		self.optimizer = [None]*2;
		self.optimizer[0] = optim.Adam(self.model[0].parameters(), lr=self.learning_rate)
		self.optimizer[1] = optim.Adam(self.model[1].parameters(), lr=self.learning_rate)

		# optim.Adam(self.model[0].parameters(), lr=self.learning_rate)
		# self.optimizer = optim.Adam(self.model[0].parameters(), lr=self.learning_rate)

		self.max_iteration = 50000

		self.w_entropy = 0.0001

		self.sum_return = 0.0

		self.loss_imitation = [0.0, 0.0]
		self.min_return_epoch = 1
		self.min_return = 10000000.0

		self.sum_loss = [0.0, 0.0]

		self.losses = [[], []]

		self.tic = time.time()

		self.episodes = [[HumanEpisodeBuffer() for y in range(self.num_agents)] for x in range(self.num_slaves)]

		self.env.resets()



	def saveModel(self):
		for i in range(2):
			self.model[i].save('../nn/current_'+str(i)+'.pt')

			if self.min_return_epoch == self.num_evaluation:
				self.model[i].save('../nn/max_'+str(i)+'.pt')
			if self.num_evaluation%20 == 0:
				self.model[i].save('../nn/'+str(self.num_evaluation)+'_'+str(i)+'.pt')


	def loadModel(self,path,index):
		self.model[index].load('../nn/'+path+'_'+str(index)+'.pt')


	# def getHardcodedAction(self, slave_index, agent_index):
	# 	return np.array([0,0,-1])


	def generateTransitions(self):
		self.total_episodes = []
		self.num_episode = 0

		states = [None]*self.num_slaves*self.num_agents
		actions = [None]*self.num_slaves*self.num_agents

		terminated = [False]*self.num_slaves*self.num_agents

		for i in range(self.num_slaves):
			for j in range(self.num_agents):
				states[i*self.num_agents+j] = self.env.getLocalState(i,j)
				actions[i*self.num_agents+j] = self.env.getHardcodedAction(i,j)


		local_step = 0
		counter = 0

		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')

			for i in range(self.num_slaves):
				for j in range(self.num_agents):
					self.env.setAction(actions[i*self.num_agents+j], i, j);

			self.env.stepsAtOnce()

			for i in range(self.num_slaves):
				nan_occur = False
				terminated_state = True
				for k in range(self.num_agents):
					self.env.getReward(i, k)
					if np.any(np.isnan(states[i*self.num_agents+k])) or np.any(np.isnan(actions[i*self.num_agents+k])):
						nan_occur = True

				if nan_occur is True:
					# for k in range(self.num_agents):
						# self.total_episodes.append(self.episodes[i][k])
						# self.episodes[i][k] = RNNEpisodeBuffer()
					self.num_episode += 1
					self.env.reset(i)


				if self.env.isTerminalState(i) is False:
					terminated_state = False
					for k in range(self.num_agents):
						self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k])
						local_step += 1

				if terminated_state is True:
					for k in range(self.num_agents):
						self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k])
						# self.total_episodes.append(self.episodes[i][k])
						# self.episodes[i][k] = HumanEpisodeBuffer()
					self.num_episode += 1
					self.env.reset(i)

			if local_step >= self.buffer_size:
				for i in range(self.num_slaves):
					self.num_episode += 1
					# self.total_episodes.append(self.episodes[i][k])
					# self.episodes[i][k] = HumanEpisodeBuffer()
					self.env.reset(i)
				break

			# get the updated state and updated hidden
			for i in range(self.num_slaves):
				for j in range(self.num_agents):
					states[i*self.num_agents+j] = self.env.getLocalState(i,j)
					actions[i*self.num_agents+j] = self.env.getHardcodedAction(i,j)

		self.env.endOfIteration()
		print('SIM : {}'.format(local_step))


	def optimizeSimulationNN(self):
		# exit(0);
		self.sum_loss = [0.0, 0.0]
		replay_buffer = [None]*2
		replay_buffer[0] = HumanEpisodeBuffer()
		replay_buffer[1] = HumanEpisodeBuffer()

		for i in range(self.num_slaves):
			for j in range(self.num_agents):
				data = self.episodes[i][j].getData()
				for k in range(len(data)):
					replay_buffer[j%2].data.append(data[k])

		self.num_tuple = self.batch_size * (len(replay_buffer[0].getData())//self.batch_size)
		# print(len(replay_buffer))
		# self.num_episode = 
		# replay_buffer[0] = np.array(self.episodes[])

		for t in range(2):
			replay_buffer_array = replay_buffer[t].getData()
			for j in range(self.num_epochs):
				# np.random.shuffle(replay_buffer_array)

				for i in range(len(replay_buffer_array)//self.batch_size):
					batch_transitions = replay_buffer_array[i*self.batch_size:(i+1)*self.batch_size]

					batch = SimpleTransition(*zip(*batch_transitions))

					stack_s = np.vstack(batch.s).astype(np.float32)
					stack_a = np.vstack(batch.a).astype(np.float32)

					cur_a = self.model[t].forward(Tensor(stack_s))	



					'''Actor Loss'''
					# ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
					# stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+1E-5)
					# stack_gae = Tensor(stack_gae)
					# surrogate1 = ratio * stack_gae
					# surrogate2 = torch.clamp(ratio, min=1.0-self.clip_ratio, max=1.0+self.clip_ratio) * stack_gae

					# loss_actor = - torch.min(surrogate1, surrogate2).mean()
					# # loss_actor = - surrogate2.mean()

					# '''Entropy Loss'''
					# loss_entropy = - self.w_entropy * a_dist.entropy().mean()
					if j == 0 and i == 0 :
						print(cur_a)
						print(stack_a)
						print("########")
						print(cur_a - Tensor(stack_a))
						print((cur_a - Tensor(stack_a)).mean())
						print("-------------------------------------`")

					loss_scale = 20.0
					loss_target = loss_scale *5.0*(((cur_a - Tensor(stack_a))/1.0).pow(2)).mean()
					loss_reg = loss_scale * 5.0*(cur_a).pow(2).mean()

					# print(self.loss_target)
					# exit(0)

					loss = loss_target + 0.01 * loss_reg
					# print(loss)
					self.optimizer[t].zero_grad()


					# start = time.time()
					loss.backward(retain_graph=True)
					# print("time :", time.time() - start)

					for param in self.model[t].parameters():
						if param.grad is not None:
							param.grad.data.clamp_(-0.5, 0.5)
					self.optimizer[t].step()
					self.loss_imitation[t] = loss.cpu().detach().numpy().tolist()
					self.sum_loss[t] += self.batch_size * self.loss_imitation[t]/self.num_epochs
			print('Optimizing simulation nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')

		for i in range(self.num_slaves):
			for k in range(self.num_agents):
				self.episodes[i][k] = HumanEpisodeBuffer()



	def optimizeModel(self):
		self.optimizeSimulationNN()


	def train(self):
		frac = 1.0
		self.learning_rate = self.default_learning_rate*frac
		self.clip_ratio = self.default_clip_ratio*frac
		for i in range(2):
			for param_group in self.optimizer[i].param_groups:
				param_group['lr'] = self.learning_rate
		self.generateTransitions();
		# self.generateHindsightTransitions();
		self.optimizeModel()
	# def loadModel(self,path,index):
	# 	self.model[index].load('../nn/'+path+"_sc.pt")
		# self.lactor_model[index].load('../nn/'+path+"_la.pt")

	def evaluate(self):
		self.num_evaluation = self.num_evaluation + 1
		h = int((time.time() - self.tic)//3600.0)
		m = int((time.time() - self.tic)//60.0)
		s = int((time.time() - self.tic))
		m = m - h*60
		s = int((time.time() - self.tic))
		s = s - h*3600 - m*60
		if self.num_episode is 0:
			self.num_episode = 1
		if self.num_tuple is 0:
			self.num_tuple = 1
		if self.min_return > self.sum_loss[0]/self.num_tuple:
			self.min_return = self.sum_loss[0]/self.num_tuple
			self.min_return_epoch = self.num_evaluation

		# if self.min_return > self.sum_loss[0]/self.num_tuple:
		# 	self.min_return = self.sum_loss[0]/self.num_tuple
		# 	self.min_return_epoch = self.num_evaluation
		self.num_tuple_so_far += self.num_tuple


		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,h,m,s))
		print('||--------------SimulationNN------------------')
		# print('||Loss Actor               : {:.4f}'.format(self.loss_actor))
		# print('||Loss Critic              : {:.4f}'.format(self.loss_critic))
		print('||Loss Imitation-DEF              : {:.4f}'.format(self.loss_imitation[0]))
		print('||Loss Imitation-ATK              : {:.4f}'.format(self.loss_imitation[1]))
		# print('||Noise                    : {:.3f}'.format(self.model[0].log_std.exp().mean()))		
		print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
		print('||Num Transition           : {}'.format(self.num_tuple))
		print('||Num Episode              : {}'.format(self.num_episode))
		print('||Avg Avg Loss tuple-DEF   : {:.3f}'.format(1.0 * self.sum_loss[0]/self.num_tuple))
		print('||Avg Avg Loss tuple-ATK   : {:.3f}'.format(1.0 * self.sum_loss[1]/self.num_tuple))
		# print('||Avg Reward per transition: {:.3f}'.format(self.sum_return/self.num_tuple))
		# print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
		# print('||MIN Avg Loss So far     : {:.3f} at #{}'.format(self.min_loss,self.max_return_epoch))
		# print('||-----------------LActorNN------------------')
		# print('||Loss Actor               : {:.4f}'.format(self.lactor_loss_actor))
		# print('||Loss Critic              : {:.4f}'.format(self.lactor_loss_critic))
		# print('||Noise                    : {:.3f}'.format(self.lactor_model[0].log_std.exp().mean()))		
		print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
		print('||Num Transition           : {}'.format(self.num_tuple))
		# print('||Num Episode              : {}'.format(self.num_episode))
		# print('||Avg Return per episode   : {:.3f}'.format(self.lactor_sum_return/self.num_episode))
		# print('||Avg Reward per transition: {:.3f}'.format(self.lactor_sum_return/self.num_tuple))
		# print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
		print('||Min Avg Loss DEF So far     : {:.3f} at #{}'.format(self.min_return, self.min_return_epoch))
		self.losses[0].append(1.0 * self.sum_loss[0]/self.num_tuple)
		self.losses[1].append(1.0 * self.sum_loss[1]/self.num_tuple)
		# self.lactor_rewards.append(self.lactor_sum_return/self.num_episode)
		
		self.saveModel()
		
		print('=============================================')
		return np.array(self.losses[0]), np.array(self.losses[1])


import matplotlib
import matplotlib.pyplot as plt

plt.ion()


def plot(y,title,num_fig=1,ylim=True):
	temp_y = np.zeros(y.shape)
	if y.shape[0]>5:
		temp_y[0] = y[0]
		temp_y[1] = 0.5*(y[0] + y[1])
		temp_y[2] = 0.3333*(y[0] + y[1] + y[2])
		temp_y[3] = 0.25*(y[0] + y[1] + y[2] + y[3])
		for i in range(4,y.shape[0]):
			temp_y[i] = np.sum(y[i-4:i+1])*0.2

	fig = plt.figure(num_fig)

	plt.clf()
	plt.title(title)
	plt.plot(y,'b')
	
	plt.plot(temp_y,'r')
	plt.axhline(y=1, color='r', linewidth=1)

	# plt.show()
	if ylim:
		plt.ylim([0,1])

	fig.canvas.draw()
	fig.canvas.flush_events()
	# plt.pause(0.001)


import argparse
if __name__=="__main__":
	il = ImitationLearning()
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')
	parser.add_argument('-iter','--iteration',help='num iterations')
	parser.add_argument('-n','--name',help='name of training setting')

	args =parser.parse_args()
	
	graph_name = ''
	
	if args.model is not None:
		il.loadModel(args.model, 0)
		il.loadModel(args.model, 1)
		if args.iteration is not None:
			il.num_evaluation = int(args.iteration)
			for i in range(int(args.iteration)):
				il.env.endOfIteration()
	if args.name is not None:
		graph_name = args.name

	else:
		il.saveModel()
	print('num states: {}, num actions: {}'.format(il.env.getNumState(),il.env.getNumAction()))
	# for i in range(ppo.max_iteration-5):
	for i in range(5000000):
		il.train()
		# rewards, lactor_rewards = slac.evaluate()
		rewards_def, rewards_atk = il.evaluate()
		plot(rewards_def, graph_name + ' def imitation learning loss per 10 tuple',0,False)
		plot(rewards_atk, graph_name + ' atk imitation learning loss per 10 tuple',1,False)

		# plot(lactor_rewards,'lactor reward',1,False)
