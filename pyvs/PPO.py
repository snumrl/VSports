import math
import random
import time
import os
import sys
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
Episode = namedtuple('Episode', ('s','a','r','value','logprob'))

class EpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def push(self, *args):
		self.data.append(Episode(*args))

	def getData(self):
		return self.data

Transition = namedtuple('Transition',('s','a','logprob','TD','GAE'))
class ReplayBuffer(object):
	def __init__(self, buff_size = 10000):
		super(ReplayBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def push(self,*args):
		self.buffer.append(Transition(*args))

	def clear(self):
		self.buffer.clear()

class PPO(object):
	def __init__(self):
		np.random.seed(seed = int(time.time()))
		self.env = Env(600)
		self.num_slaves = 16
		self.num_agents = 2
		self.num_state = self.env.getNumState()
		self.num_action = self.env.getNumAction()

		self.num_epochs = 10
		self.num_epochs_muscle = 3
		self.num_evaluation = 0
		self.num_tuple_so_far = 0
		self.num_episode = 0
		self.num_tuple = 0
		self.num_simulation_Hz = self.env.getSimulationHz()
		self.num_control_Hz = self.env.getControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.95
		self.lb = 0.95

		self.buffer_size = 2*2048
		self.batch_size = 128
		self.muscle_batch_size = 128
		self.replay_buffer = ReplayBuffer(30000)

		self.model = SimulationNN(self.num_state, self.num_action)
		if use_cuda:
			self.model.cuda()

		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
		self.max_iteration = 50000

		self.w_entropy = 0.0001

		self.loss_actor = 0.0
		self.loss_critic = 0.0
		self.rewards = []
		self.sum_return = 0.0
		self.max_return = 1.0
		self.max_return_epoch = 1
		self.tic = time.time()

		self.episodes = [None]*self.num_slaves*self.num_agents
		for j in range(self.num_slaves*self.num_agents):
			self.episodes[j] = EpisodeBuffer()
		self.env.resets()

	def saveModel(self):
		self.model.save('../nn/current.pt')

		if self.max_return_epoch == self.num_evaluation:
			self.model.save('../nn/max.pt')
		if self.num_evaluation%100 == 0:
			self.model.save('../nn/'+str(self.num_evaluation//100)+'.pt')

	def loadModel(self,path):
		self.model.load('../nn/'+path+'.pt')

	def computeTDandGAE(self):
		self.replay_buffer.clear();
		self.sum_return = 0.0
		for epi in self.total_episodes:
			data = epi.getData()
			size = len(data)
			if size == 0:
				continue
			# print("Size : ",size)
			states, actions, rewards, values, logprobs = zip(*data)
			values = np.concatenate((values, np.zeros(1)), axis=0)
			advantages = np.zeros(size)
			ad_t = 0

			epi_return = 0.0
			for i in reversed(range(len(data))):
				epi_return += rewards[i]
				delta = rewards[i] + values[i+1] * self.gamma - values[i]
				ad_t = delta + self.gamma * self.lb * ad_t
				advantages[i] = ad_t
			self.sum_return += epi_return
			TD = values[:size] + advantages

			for i in range(size):
				self.replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

		self.num_episode = len(self.total_episodes)
		self.num_tuple = len(self.replay_buffer.buffer)
		print('SIM : {}'.format(self.num_tuple))
		self.num_tuple_so_far += self.num_tuple

	def generateTransitions(self):
		self.total_episodes = []
		states = [None]*self.num_slaves*self.num_agents
		actions = [None]*self.num_slaves*self.num_agents
		rewards = [None]*self.num_slaves*self.num_agents
		states_next = [None]*self.num_slaves*self.num_agents
		for i in range(self.num_slaves):
			for j in range(self.num_agents):
				states[i*self.num_agents+j] = self.env.getState(i,j)
		local_step = 0
		terminated = [False]*self.num_slaves*self.num_agents
		counter = 0
		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')
			a_dist,v = self.model(Tensor(states))
			# print(self.model)
			# print(self)
			# print(a_dist)
			# print()
			# print(v)
			# exit()
			actions = a_dist.sample().cpu().detach().numpy()

			logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
			values = v.cpu().detach().numpy().reshape(-1)
			for i in range(self.num_slaves):
				for j in range(self.num_agents):
					self.env.setAction(actions[i*self.num_agents+j], i, j);

			self.env.stepsAtOnce()
			# self.env.step(0)

			for j in range(self.num_slaves):
				nan_occur = False
				terminated_state = True

				if np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j])):
					nan_occur = True

				elif self.env.isTerminalState(j) is False:
					terminated_state = False
					for k in range(self.num_agents):
						rewards[j*self.num_agents+k] = self.env.getReward(j, k)
						self.episodes[j*self.num_agents+k].push(states[j*self.num_agents+k], actions[j*self.num_agents+k],\
							rewards[j*self.num_agents+k], values[j*self.num_agents+k], logprobs[j*self.num_agents+k])
						local_step += 1

				if nan_occur is True:
					for k in range(self.num_agents):
						self.total_episodes.append(self.episodes[j*self.num_agents+k])
						self.episodes[j*self.num_agents+k] = EpisodeBuffer()
					self.env.reset(j)

				elif terminated_state :
					for k in range(self.num_agents):
						self.total_episodes.append(self.episodes[j*self.num_agents+k])
						self.episodes[j*self.num_agents+k] = EpisodeBuffer()
					self.env.reset(j)



			if local_step >= self.buffer_size:
				for k in range(self.num_agents):
					self.total_episodes.append(self.episodes[j*self.num_agents+k])
					self.episodes[j*self.num_agents+k] = EpisodeBuffer()
				self.env.reset(j)
				break
			for i in range(self.num_slaves):
				for j in range(self.num_agents):
					states[i*self.num_agents+j] = self.env.getState(i,j)


	def optimizeSimulationNN(self):
		all_transitions = np.array(self.replay_buffer.buffer)
		for j in range(self.num_epochs):
			np.random.shuffle(all_transitions)
			for i in range(len(all_transitions)//self.batch_size):
				transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
				batch = Transition(*zip(*transitions))

				stack_s =np.vstack(batch.s).astype(np.float32)
				stack_a = np.vstack(batch.a).astype(np.float32)
				stack_lp = np.vstack(batch.logprob).astype(np.float32)
				stack_td = np.vstack(batch.TD).astype(np.float32)
				stack_gae = np.vstack(batch.GAE).astype(np.float32)

				a_dist,v = self.model(Tensor(stack_s))

				'''Critic Loss'''
				loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()

				'''Actor Loss'''
				ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
				stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+1E-5)
				stack_gae = Tensor(stack_gae)
				surrogate1 = ratio * stack_gae
				surrogate2 = torch.clamp(ratio, min=1.0-self.clip_ratio, max=1.0+self.clip_ratio) * stack_gae
				loss_actor = - torch.min(surrogate1, surrogate2).mean()

				'''Entropy Loss'''
				loss_entropy = - self.w_entropy * a_dist.entropy().mean()

				self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
				self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

				loss = loss_actor + loss_entropy + loss_critic

				self.optimizer.zero_grad()
				loss.backward(retain_graph=True)
				for param in self.model.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5, 0.5)
				self.optimizer.step()
			print('Optimizing sim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')


	def optimizeModel(self):
		self.computeTDandGAE()
		self.optimizeSimulationNN()

	def train(self):
		frac = 1.0
		self.learning_rate = self.default_learning_rate*frac
		self.clip_ratio = self.default_clip_ratio*frac
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.learning_rate

		self.generateTransitions();
		self.optimizeModel()



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
		if self.max_return < self.sum_return/self.num_episode:
			self.max_return = self.sum_return/self.num_episode
			self.max_return_epoch = self.num_evaluation
		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,h,m,s))
		print('||Loss Actor               : {:.4f}'.format(self.loss_actor))
		print('||Loss Critic              : {:.4f}'.format(self.loss_critic))
		print('||Noise                    : {:.3f}'.format(self.model.log_std.exp().mean()))		
		print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
		print('||Num Transition           : {}'.format(self.num_tuple))
		print('||Num Episode              : {}'.format(self.num_episode))
		print('||Avg Return per episode   : {:.3f}'.format(self.sum_return/self.num_episode))
		print('||Avg Reward per transition: {:.3f}'.format(self.sum_return/self.num_tuple))
		print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
		print('||Max Avg Retun So far     : {:.3f} at #{}'.format(self.max_return,self.max_return_epoch))
		self.rewards.append(self.sum_return/self.num_episode)
		
		self.saveModel()
		
		print('=============================================')
		return np.array(self.rewards)

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

	plt.figure(num_fig)
	plt.clf()
	plt.title(title)
	plt.plot(y,'b')
	
	plt.plot(temp_y,'r')

	plt.show()
	if ylim:
		plt.ylim([0,1])
	plt.pause(0.001)

import argparse

if __name__=="__main__":
	ppo = PPO()
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')

	args =parser.parse_args()
	
	
	if args.model is not None:
		ppo.loadModel(args.model)
	else:
		ppo.saveModel()
	print('num states: {}, num actions: {}'.format(ppo.env.getNumState(),ppo.env.getNumAction()))
	for i in range(ppo.max_iteration-5):
		ppo.train()
		rewards = ppo.evaluate()
		plot(rewards,'reward',0,False)





