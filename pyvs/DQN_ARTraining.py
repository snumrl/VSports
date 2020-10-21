import math
import random
import time
import os
import sys
import copy
from datetime import datetime

from os.path import join, exists
from os import mkdir

import collections
from collections import namedtuple
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from Utils import RunningMeanStd
from VAE import VAEDecoder

import numpy as np

from pyvs import Env

from IPython import embed
import json
from Model import *
from pathlib import Path


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")


nnCount = 0
baseDir = "../nn_dqn_lar_h"
nndir = baseDir + "/nn"+str(nnCount)

if not exists(baseDir):
    mkdir(baseDir)

if not exists(nndir):
	mkdir(nndir)


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

class Buffer(object):
	def __init__(self, buff_size = 10000):
		super(Buffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def push(self,replay_buffer):
		self.buffer.append(replay_buffer)

	def clear(self):
		self.buffer.clear()


DQNEpisode = namedtuple('DQNEpisode', ('s','a','r','s1','terminated'))

class DQNEpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def push(self, *args):
		self.data.append(DQNEpisode(*args))

	def pushEpisode(self, epi):
		self.data.append(epi)

	def getData(self):
		return self.data

	def sample(self, batch_size):
		# embed()
		# exit(0)
		state, action, reward, next_state, terminated = zip(*random.sample(self.data, batch_size))
		return state, action, reward, next_state, terminated
    
# DQNTransition = namedtuple('DQNTransition',('s','a'))

# class ReplayBuffer(object):
# 	def __init__(self, buff_size = 10000):
# 		super(ReplayBuffer, self).__init__()
# 		self.buffer = deque(maxlen=buff_size)

# 	def push(self,*args):
# 		self.buffer.append(Transition(*args))

# 	def clear(self):
# 		self.buffer.clear()

# class Buffer(object):
# 	def __init__(self, buff_size = 10000):
# 		super(Buffer, self).__init__()
# 		self.buffer = deque(maxlen=buff_size)

# 	def push(self,replay_buffer):
# 		self.buffer.append(replay_buffer)

# 	def clear(self):
# 		self.buffer.clear()


class RL(object):
	def __init__(self, motion_nn):
		np.random.seed(seed = int(time.time()))
		self.num_slaves = 16
		self.num_agents = 1
		self.env = Env(self.num_agents, motion_nn, self.num_slaves)
		self.num_state = self.env.getNumState()

		self.num_policy = 1

		self.num_epochs = 4	
		self.num_evaluation = 0
		self.num_tuple_so_far = [0, 0]
		self.num_tuple = [0, 0]

		self.num_simulation_Hz = self.env.getSimulationHz()
		self.num_control_Hz = self.env.getControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.997
		self.lb = 0.95

		self.buffer_size = 4*1024
		self.batch_size = 256

		self.num_action_types = 2
		self.latent_size = 4

		self.ppo_buffer = [ [None] for _ in range(self.num_policy)]
		
		for i in range(self.num_policy):
			self.ppo_buffer[i] = Buffer(100000)

		self.actionDecoders = [ VAEDecoder().to(device) for _ in range(self.num_action_types)]

		self.actionDecoders[0].load("vae_nn4/vae_action_decoder_"+str(0)+".pt")
		self.actionDecoders[1].load("vae_nn4/vae_action_decoder_"+str(3)+".pt")

		self.rms = RunningMeanStd(self.num_state)


		self.dqn_model = [[None] for _ in range(self.num_policy)]
		for i in range(self.num_policy):
			self.dqn_model[i] = DQN(self.num_state, self.num_action_types)
			if use_cuda:
				self.dqn_model[i].cuda()


		self.ppo_model = [[None] for _ in range(self.num_policy)]
		for i in range(self.num_policy):
			self.ppo_model[i] = ActorCriticNN(self.num_state + self.num_action_types, self.latent_size, 0.0)
			if use_cuda:
				self.ppo_model[i].cuda()



		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio

		self.dqn_optimizer = [[None] for _ in range(self.num_policy)]
		self.ppo_optimizer = [[None] for _ in range(self.num_policy)]

		for i in range(self.num_policy):
			self.dqn_optimizer[i] = optim.Adam(self.dqn_model[i].parameters(), lr=self.learning_rate)

		for i in range(self.num_policy):
			self.ppo_optimizer[i] = optim.Adam(self.ppo_model[i].parameters(), lr=self.learning_rate)



		self.max_iteration = 50000

		self.w_entropy = 0.0001

		self.ppo_sum_loss_actor = [0.0 for _ in range(self.num_policy)] 
		self.ppo_sum_loss_critic = [0.0 for _ in range(self.num_policy)]

		self.dqn_sum_loss = [0.0 for _ in range(self.num_policy)]

		self.rewards = []

		self.sum_return = 0.0
		self.max_return = 0.0

		self.max_return_epoch = 1

		self.tic = time.time()

	
		self.ppo_episodes = [[EpisodeBuffer() for _ in range(self.num_agents)] for _ in range(self.num_slaves)]
		self.dqn_episodes = [[DQNEpisodeBuffer() for _ in range(self.num_agents)] for _ in range(self.num_slaves)]

		self.indexToNetDic = {0:0, 1:0}

		self.filecount = 0

		self.env.slaveResets()

		self.rms = RunningMeanStd(self.num_state)

	def saveModels(self):
		for i in range(self.num_policy):
			self.ppo_model[i].save(nndir+'/'+'current_'+str(i)+'.pt')


			if self.max_return_epoch == self.num_evaluation:
				self.ppo_model[i].save(nndir+'/'+'max_'+str(i)+'.pt')

			if self.num_evaluation%100 == 0:
				self.ppo_model[i].save(nndir+'/'+str(self.num_evaluation)+'_'+str(i)+'.pt')
		self.rms.save(nndir+'/rms.ms')


	def generateTransitions(self):
		self.total_ppo_episodes = [[] for i in range(self.num_policy)]
		self.total_dqn_episodes = [[] for i in range(self.num_policy)]

		dqn_states = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]
		dqn_actions = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]
		dqn_nextStates = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]
		dqn_next_states = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]

		states = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]
		actions = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]
		rewards = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]
		logprobs = np.array([None for _ in range(self.num_agents)])
		values = np.array([None for _ in range(self.num_agents)])

		terminated = [False]*self.num_slaves*self.num_agents

		for i in range(self.num_agents):
			for j in range(self.num_slaves):
				dqn_states[i][j] = self.env.getState(j,i).astype(np.float32)
		dqn_states = np.array(dqn_states)
		dqn_states = self.rms.apply(dqn_states)

		terminated = [False]*self.num_slaves*self.num_agents

		learningTeam = 0
		teamDic = {0: 0, 1: 0}

		local_step = 0
		counter = 0


		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')


			'''DQN transition'''
			epsilon = 0.5 * pow(0.997, self.num_evaluation);

			for i in range(self.num_agents):
				dqn_actions = self.dqn_model[self.indexToNetDic[i]].act(dqn_states, epsilon)

			dqn_actions_one_hot = torch.nn.functional.one_hot(dqn_actions, num_classes=2).cpu().detach().numpy()
			# embed()
			# exit(0)



			'''PPO transition'''
			# embed()
			# exit(0)
			# print(len(dqn_states[0][0]))
			# print(len(dqn_actions_one_hot[0][0]))
			states = np.concatenate((dqn_states, dqn_actions_one_hot),axis=2)
			# print(len(states[0][0]))
			# print("###########")
			
			a_dist_slave = [None]*self.num_agents
			v_slave = [None]*self.num_agents
			for i in range(self.num_agents):
				if teamDic[i] == learningTeam:
					a_dist_slave_agent,v_slave_agent = self.ppo_model[self.indexToNetDic[i]].forward(\
						Tensor(states[i]))
					a_dist_slave[i] = a_dist_slave_agent
					v_slave[i] = v_slave_agent
					actions[i] = a_dist_slave[i].sample().cpu().detach().numpy().squeeze().squeeze();		

			for i in range(self.num_agents):
				if teamDic[i] == learningTeam:
					logprobs[i] = a_dist_slave[i].log_prob(Tensor(actions[i]))\
						.cpu().detach().numpy().reshape(-1);
					values[i] = v_slave[i].cpu().detach().numpy().reshape(-1);
			
			# embed()
			# exit(0)
			# actions = actions.astype(np.float32)

			decodeShape = list(np.shape(actions))
			decodeShape[2] = 9
			actionsDecoded =np.empty(decodeShape,dtype=np.float32)

			# print(dqn_actions)


			for i in range(self.num_agents):
				for j in range(self.num_slaves):
					curActionType = dqn_actions[i][j].item()
					# embed()
					# exit(0)
					self.env.setActionType(curActionType, j, i);
					# embed()
					# exit(0)
					actionsDecoded[i][j] = self.actionDecoders[curActionType].decode(Tensor(actions[i][j])).cpu().detach().numpy()

			envActions = actionsDecoded
			for i in range(self.num_agents):
				for j in range(self.num_slaves):
					self.env.setAction(envActions[i][j], j, i);

			self.env.stepsAtOnce()

			for i in range(self.num_agents):
				for j in range(self.num_slaves):
					dqn_next_states[i][j] = self.env.getState(j,i).astype(np.float32)
			dqn_next_states = np.array(dqn_next_states)
			dqn_next_states = self.rms.apply(dqn_next_states)

			nan_occur = [False]*self.num_slaves

			for j in range(self.num_slaves):
				if not self.env.isOnResetProcess(j):
					for i in range(self.num_agents):
						if teamDic[i] == learningTeam:
							rewards[i][j] = self.env.getReward(j, i, True)
							if np.any(np.isnan(rewards[i][j])):
								nan_occur[j] = True
							if np.any(np.isnan(states[i][j])) or np.any(np.isnan(actions[i][j])):
								nan_occur[j] = True

			for j in range(self.num_slaves):
				if not self.env.isOnResetProcess(j):
					if nan_occur[j] is True:
						for i in range(self.num_agents):
							if teamDic[i] == learningTeam:
								self.total_ppo_episodes[self.indexToNetDic[i]].append(self.ppo_episodes[j][i])
								# embed()
								# exit(0)
								self.dqn_episodes[j][i][-1].terminated = 1
								self.total_dqn_episodes[self.indexToNetDic[i]].append(self.dqn_episodes[j][i])
								self.ppo_episodes[j][i] = EpisodeBuffer()
								self.dqn_episodes[j][i] = DQNEpisodeBuffer()

						print("nan", file=sys.stderr)
						self.env.slaveReset(j)

					if self.env.isTerminalState(j) is False:
						for i in range(self.num_agents):
							if teamDic[i] == learningTeam:
								self.ppo_episodes[j][i].push(states[i][j], actions[i][j],\
									rewards[i][j], values[i][j], logprobs[i][j])
								self.dqn_episodes[j][i].push(dqn_states[i][j], dqn_actions[i][j],\
									rewards[i][j], dqn_next_states[i][j], 0)
								local_step += 1
					else:
						for i in range(self.num_agents):
							if teamDic[i] == learningTeam:
								self.ppo_episodes[j][i].push(states[i][j], actions[i][j],\
									rewards[i][j], values[i][j], logprobs[i][j])
								self.dqn_episodes[j][i].push(dqn_states[i][j], dqn_actions[i][j],\
									rewards[i][j], dqn_next_states[i][j], 1)

								self.total_ppo_episodes[self.indexToNetDic[i]].append(self.ppo_episodes[j][i])
								self.total_dqn_episodes[self.indexToNetDic[i]].append(self.dqn_episodes[j][i])
								self.ppo_episodes[j][i] = EpisodeBuffer()
								self.dqn_episodes[j][i] = DQNEpisodeBuffer()
						self.env.slaveReset(j)



			if local_step >= self.buffer_size:
				for j in range(self.num_slaves):
					for i in range(self.num_agents):
						if teamDic[i] == learningTeam:
							self.total_ppo_episodes[self.indexToNetDic[i]].append(self.ppo_episodes[j][i])
							self.total_dqn_episodes[self.indexToNetDic[i]].append(self.dqn_episodes[j][i])
							self.ppo_episodes[j][i] = EpisodeBuffer()
							self.dqn_episodes[j][i] = DQNEpisodeBuffer()

					self.env.slaveReset(j)
				break

			for i in range(self.num_agents):
				for j in range(self.num_slaves):
					dqn_states[i][j] = self.env.getState(j,i).astype(np.float32)
			dqn_states = np.array(dqn_states)
			dqn_states = self.rms.apply(dqn_states)
		print('SIM : {}'.format(local_step))

# DQNEpisode = namedtuple('DQNEpisode', ('s','a','r','s1','terminated'))

	def dqn_computeTDandOptimize(self):
		for index in range(self.num_policy):
			self.dqn_sum_loss[index] = 0
		gamma = 0.99

		dqn_replay_buffer = [DQNEpisodeBuffer() for _ in range(self.num_policy)]

		for index in range(self.num_policy):
			for i in range(len(self.total_dqn_episodes[index])):
				for epi in range(len(self.total_dqn_episodes[index][i].data)):
					dqn_replay_buffer[index].pushEpisode(self.total_dqn_episodes[index][i].data[epi])

				# dqn_replay_buffer[i].data.append
				# embed()
				# exit(0)(self.total_dqn_episodes[index][i])

		# for epoch in range(self.num_epochs):
		for index in range(self.num_policy):
			state, action, reward, next_state, terminated = dqn_replay_buffer[index].sample(self.batch_size)

			state      = Variable(torch.FloatTensor(np.float32(state)))
			next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=True)
			action     = Variable(torch.LongTensor(action))
			reward     = Variable(torch.FloatTensor(reward))
			terminated = Variable(torch.FloatTensor(terminated))


			# embed()
			# exit(0)
			q_values      = self.dqn_model[index](state)
			next_q_values = self.dqn_model[index](next_state)

			q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
			next_q_value     = next_q_values.max(1)[0]
			expected_q_value = reward + gamma * next_q_value * (1 - terminated)

			loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
			    
			self.dqn_optimizer[index].zero_grad()
			loss.backward()
			self.dqn_optimizer[index].step()
			self.dqn_sum_loss[index] += loss.item()*self.batch_size/self.num_epochs




	def ppo_computeTDandGAE(self):
		for index in range(self.num_policy):
			self.ppo_buffer[index].clear()
			if index == 0:
				self.sum_return = 0.0
			for epi in self.total_ppo_episodes[index]:
				data = epi.getData()
				size = len(data)
				if size == 0:
					continue
				states, actions, rewards, values, logprobs = zip(*data)
				values = np.concatenate((values, np.zeros(1)), axis=0)
				advantages = np.zeros(size)
				ad_t = 0

				epi_return = 0.0
				# embed()
				# exit(0)
				for i in reversed(range(len(data))):
					epi_return += rewards[i]
					delta = rewards[i] + values[i+1] * self.gamma - values[i]
					ad_t = delta + self.gamma * self.lb * ad_t
					advantages[i] = ad_t

				if not np.isnan(epi_return):
					if index == 0:
						self.sum_return += epi_return

					TD = values[:size] + advantages


					replay_buffer = ReplayBuffer(10000)
					for i in range(size):

						replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

					self.ppo_buffer[index].push(replay_buffer)

		''' counting numbers '''
		for index in range(self.num_policy):
			self.num_episode = len(self.total_ppo_episodes[0])
			self.num_tuple[index] = 0
			for replay_buffer in self.ppo_buffer[index].buffer:
				self.num_tuple[index] += len(replay_buffer.buffer)
			self.num_tuple_so_far[index] += self.num_tuple[index]


	def ppo_optimizeNN(self):
		# embed()
		# exit(0)
		for i in range(self.num_policy):
			self.ppo_sum_loss_actor[i] = 0.0
			self.ppo_sum_loss_critic[i] = 0.0

		for buff_index in range(self.num_policy):
			all_rnn_replay_buffer= np.array(self.ppo_buffer[buff_index].buffer)
			for j in range(self.num_epochs):
				all_segmented_transitions = []
				for rnn_replay_buffer in all_rnn_replay_buffer:
					rnn_replay_buffer_size = len(rnn_replay_buffer.buffer)
					for i in range(rnn_replay_buffer_size):
						all_segmented_transitions.append(rnn_replay_buffer.buffer[i])

				np.random.shuffle(all_segmented_transitions)
				for i in range(len(all_segmented_transitions)//self.batch_size):
					batch_segmented_transitions = all_segmented_transitions[i*self.batch_size:(i+1)*self.batch_size]

					loss = Tensor(torch.zeros(1).cuda())

					batch = Transition(*zip(*batch_segmented_transitions))

					stack_s = np.vstack(batch.s).astype(np.float32)
					stack_a = np.vstack(batch.a).astype(np.float32)
					stack_lp = np.vstack(batch.logprob).astype(np.float32)
					stack_td = np.vstack(batch.TD).astype(np.float32)
					stack_gae = np.vstack(batch.GAE).astype(np.float32)

					a_dist,v = self.ppo_model[buff_index].forward(Tensor(stack_s))	
					
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

					loss = loss_actor + loss_critic + loss_entropy
					self.ppo_optimizer[buff_index].zero_grad()

					loss.backward(retain_graph=True)

					for param in self.ppo_model[buff_index].parameters():
						if param.grad is not None:
							param.grad.data.clamp_(-0.5, 0.5)

					self.ppo_optimizer[buff_index].step()
					self.ppo_sum_loss_actor[buff_index] += loss_actor*self.batch_size/self.num_epochs
					self.ppo_sum_loss_critic[buff_index] += loss_critic*self.batch_size/self.num_epochs

				print('Optimizing actor-critic nn_ppo : {}/{}'.format(j+1,self.num_epochs),end='\r')
			print('')





	def train(self):
		frac = 1.0

		self.learning_rate = self.default_learning_rate*frac
		self.clip_ratio = self.default_clip_ratio*frac
		for i in range(self.num_policy):
			for param_group in self.ppo_optimizer[i].param_groups:
				param_group['lr'] = self.learning_rate
			for param_group in self.dqn_optimizer[i].param_groups:
				param_group['lr'] = self.learning_rate


		self.generateTransitions()
		self.dqn_computeTDandOptimize()
		self.ppo_computeTDandGAE()
		self.optimizeModel()

	def optimizeModel(self):
		self.ppo_optimizeNN()


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
		for i in range(self.num_policy):
			if self.num_tuple[i]is 0:
				self.num_tuple[i] = 1

		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,h,m,s))
		print('||--------------ActorCriticNN------------------')

		for i in range(self.num_policy):
			print('||Avg Loss Actor PPO {}     : {:.4f}'.format(i, self.ppo_sum_loss_actor[i]/self.num_tuple[i]))
			print('||Avg Loss Critic PPO {}    : {:.4f}'.format(i, self.ppo_sum_loss_critic[i]/self.num_tuple[i]))
			print('||Noise PPO                 : {:.3f}'.format(self.ppo_model[i].log_std.exp().mean()))		

			print('||Avg Loss DQN {}           : {:.4f}'.format(i, self.dqn_sum_loss[i]/self.num_tuple[i]))
			print('||Epsilon of DQN            : {:.4f}'.format(0.5 * pow(0.997, self.num_evaluation-1)))

			print('||Num Transition So far {}  : {}'.format(i, self.num_tuple_so_far[i]))
			print('||Num Transition {}         : {}'.format(i, self.num_tuple[i]))


		print('||Num Episode               : {}'.format(self.num_episode))
		print('||Avg Return per episode    : {:.3f}'.format(self.sum_return/self.num_episode))
		for i in range(self.num_policy):
			print('||Avg Step per episode {}   : {:.1f}'.format(i, self.num_tuple[i]/self.num_episode))
		# print('||Max Win Rate So far      : {:.3f} at #{}'.format(self.max_winRate,self.max_winRate_epoch))
		# print('||Current Win Rate         : {:.3f}'.format(self.winRate[-1]))

		# print('||Avg Loss Predictor RND   : {:.4f}'.format(self.sum_loss_rnd/(self.num_tuple[0]+self.num_tuple[1])))



		self.rewards.append(self.sum_return/self.num_episode)
		
		self.saveModels()
		
		print('=============================================')
		return np.array(self.rewards)





import matplotlib
import matplotlib.pyplot as plt

plt.ion()


def plot(y,title,num_fig=1,ylim=True,path=""):
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

	plt.savefig(path, format="png")

	# plt.show()
	if ylim:
		plt.ylim([0,1])

	fig.canvas.draw()
	fig.canvas.flush_events()



import argparse
if __name__=="__main__":
	# print("111111111111")
	# exit(0)
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')
	# parser.add_argument('-p','--policy',help='pretrained pollicy path')
	parser.add_argument('-iter','--iteration',help='num iterations')
	parser.add_argument('-n','--name',help='name of training setting')
	parser.add_argument('-motion', '--motion', help='motion nn path')
	# print("222222222222")

	args =parser.parse_args()

	graph_name = ''
	
	if args.motion is None:
		print("Please specify the motion nn path")
		exit(0)

	rl = RL(args.motion)

	if args.model is not None:
		for k in range(rl.num_agents):
			rl.loadTargetModels(args.model, k)

	if args.name is not None:
		graph_name = args.name

	# if args.policy is not None:
	# 	rl.loadTargetPolicy(args.policy)
	if args.iteration is not None:
		rl.num_evaluation = int(args.iteration)
		for i in range(int(args.iteration)):
			rl.env.endOfIteration()
	else:
		rl.saveModels()


	print('num states: {}, num actions: {}'.format(rl.env.getNumState(),rl.env.getNumAction()-1))
	# for i in range(ppo.max_iteration-5):

	result_figure = nndir+"/"+"result.png"
	result_figure_num = 0
	while Path(result_figure).is_file():
		result_figure = nndir+"/"+"result_{}.png".format(result_figure_num)
		result_figure_num+=1

	for i in range(5000000):
		rl.train()
		rewards = rl.evaluate()
		plot(rewards, graph_name + 'Reward',0,False, path=result_figure)
		# plot_winrate(winRate, graph_name + 'vs Hardcoded Winrate',1,False)

