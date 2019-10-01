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

class SLAC(object):
	def __init__(self):
		np.random.seed(seed = int(time.time()))
		self.env = Env(600)
		self.num_slaves = 8
		self.num_agents = 2
		self.num_state = self.env.getNumState()
		self.num_action = self.env.getNumAction()

		self.num_epochs = 4
		self.num_evaluation = 0
		self.num_tuple_so_far = 0
		self.num_episode = 0
		self.num_tuple = 0

		self.num_simulation_Hz = self.env.getSimulationHz()
		self.num_control_Hz = self.env.getControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.997
		self.lb = 0.95

		self.buffer_size = 2*2048
		self.batch_size = 256
		self.trunc_size = 32
		self.burn_in_size = 16

		self.scheduler_buffer = Buffer(30000)
		self.lactor_buffer = Buffer(30000)

		self.scheduler_model = [None]*self.num_slaves*self.num_agents
		self.lactor_model = [None]*self.num_slaves*self.num_agents
		for i in range(self.num_slaves*self.num_agents):
			# exit()
			self.scheduler_model[i] = SchedulerNN(self.num_state, self.num_action)
			self.lactor_model[i] = LActorNN(self.num_state, self.num_action)

			if use_cuda:
				self.scheduler_model[i].cuda()
				self.lactor_model[i].cuda()

		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio
		self.scheduler_optimizer = optim.Adam(self.scheduler_model[0].parameters(), lr=self.learning_rate)
		self.lactor_optimizer = optim.Adam(self.lactor_model[0].parameters(), lr=self.learning_rate)

		self.max_iteration = 50000

		self.w_entropy = 0.0001

		self.scheduler_loss_actor = 0.0
		self.scheduler_loss_critic = 0.0
		self.lactor_loss_actor = 0.0
		self.lactor_loss_critic = 0.0

		self.scheduler_rewards = []
		self.lactor_rewards = []

		self.scheduler_sum_return = 0.0
		self.lactor_sum_return = 0.0

		self.scheduler_max_return = 0.0
		self.lactor_max_return = 0.0

		self.scheduler_max_return_epoch = 1
		self.lactor_max_return_epoch = 1
		self.tic = time.time()

		self.scheduler_episodes = [[RNNEpisodeBuffer() for y in range(self.num_agents)] for x in range(self.num_slaves)]
		self.lactor_episodes = [[RNNEpisodeBuffer() for y in range(self.num_agents)] for x in range(self.num_slaves)]

		self.env.resets()



	def saveModel(self):
		self.scheduler_model[0].save('../nn/current_sc.pt')
		self.lactor_model[0].save('../nn/current_la.pt')

		if self.scheduler_max_return_epoch == self.num_evaluation:
			self.scheduler_model[0].save('../nn/max_sc.pt')
		if self.num_evaluation%20 == 0:
			self.scheduler_model[0].save('../nn/'+str(self.num_evaluation)+'_sc.pt')

		if self.lactor_max_return_epoch == self.num_evaluation:
			self.lactor_model[0].save('../nn/max_la.pt')
		if self.num_evaluation%20 == 0:
			self.lactor_model[0].save('../nn/'+str(self.num_evaluation)+'_la.pt')


	def loadlModel(self,path,index):
		self.scheduler_model[index].load('../nn/'+path+'_sc.pt')
		self.lactor_model[index].load('../nn/'+path+'_la.pt')


	def loadlModel(self,path1,path2,index):
		self.scheduler_model[index].load('../nn/'+path1+'_sc.pt')
		self.lactor_model[index].load('../nn/'+path2+'_la.pt')


	def computeTDandGAE(self):

		'''Scheduler'''
		self.scheduler_buffer.clear()
		self.scheduler_sum_return = 0.0
		for epi in self.total_scheduler_episodes:
			data = epi.getData()
			size = len(data)
			if size == 0:
				continue
			states, actions, rewards, values, logprobs, hiddens = zip(*data)
			values = np.concatenate((values, np.zeros(1)), axis=0)
			advantages = np.zeros(size)
			ad_t = 0

			epi_return = 0.0
			for i in reversed(range(len(data))):
				epi_return += rewards[i]
				delta = rewards[i] + values[i+1] * self.gamma - values[i]
				ad_t = delta + self.gamma * self.lb * ad_t
				advantages[i] = ad_t

			if not np.isnan(epi_return):
				self.scheduler_sum_return += epi_return
				TD = values[:size] + advantages

				rnn_replay_buffer = RNNReplayBuffer(4000)
				for i in range(size):
					rnn_replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i], hiddens[i][0], hiddens[i][1])
				self.scheduler_buffer.push(rnn_replay_buffer)

		''' Linear Actor '''
		# self.lactor_buffer.clear()
		# self.lactor_sum_return = 0.0
		# for epi in self.total_lactor_episodes:
		# 	data = epi.getData()
		# 	size = len(data)
		# 	if size == 0:
		# 		continue
		# 	states, actions, rewards, values, logprobs, hiddens = zip(*data)
		# 	values = np.concatenate((values, np.zeros(1)), axis=0)
		# 	advantages = np.zeros(size)
		# 	ad_t = 0

		# 	epi_return = 0.0
		# 	for i in reversed(range(len(data))):
		# 		epi_return += rewards[i]
		# 		delta = rewards[i] + values[i+1] * self.gamma - values[i]
		# 		ad_t = delta + self.gamma * self.lb * ad_t
		# 		advantages[i] = ad_t

		# 	if not np.isnan(epi_return):
		# 		self.lactor_sum_return += epi_return
		# 		TD = values[:size] + advantages

		# 		rnn_replay_buffer = RNNReplayBuffer(4000)
		# 		for i in range(size):
		# 			rnn_replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i], hiddens[i][0], hiddens[i][1])
		# 		self.lactor_buffer.push(rnn_replay_buffer)


		''' counting numbers(w.r.t scheduler) '''
		self.num_episode = len(self.total_scheduler_episodes)
		self.num_tuple = 0
		for rnn_replay_buffer in self.scheduler_buffer.buffer:
			self.num_tuple += len(rnn_replay_buffer.buffer)
		self.num_tuple_so_far += self.num_tuple



	def getHardcodedAction(self, slave_index, agent_index):
		return np.array([0,0,-1])


	def generateTransitions(self):
		self.total_scheduler_episodes = []
		self.total_lactor_episodes = []

		scheduler_states = [None]*self.num_slaves*self.num_agents
		scheduler_actions = [None]*self.num_slaves*self.num_agents
		scheduler_rewards = [None]*self.num_slaves*self.num_agents
		scheduler_logprobs = [None]*self.num_slaves*self.num_agents
		scheduler_values = [None]*self.num_slaves*self.num_agents
		'''hiddens : (hidden ,cell) tuple''' 
		scheduler_hiddens = [None]*self.num_slaves*self.num_agents
		scheduler_hiddens_forward = [None]*self.num_slaves*self.num_agents
		scheduler_terminated = [False]*self.num_slaves*self.num_agents

		for i in range(self.num_slaves):
			for j in range(self.num_agents):
				scheduler_states[i*self.num_agents+j] = self.env.getSchedulerState(i,j)
				scheduler_hiddens[i*self.num_agents+j] = self.scheduler_model[0].init_hidden(1)
				scheduler_hiddens[i*self.num_agents+j] = (scheduler_hiddens[i*self.num_agents+j][0].cpu().detach().numpy(), \
							scheduler_hiddens[i*self.num_agents+j][1].cpu().detach().numpy())
				scheduler_hiddens_forward[i*self.num_agents+j] = self.scheduler_model[0].init_hidden(1)

		lactor_states = [None]*self.num_slaves*self.num_agents
		lactor_actions = [None]*self.num_slaves*self.num_agents
		lactor_rewards = [None]*self.num_slaves*self.num_agents
		lactor_logprobs = [None]*self.num_slaves*self.num_agents
		lactor_values = [None]*self.num_slaves*self.num_agents
		'''hiddens : (hidden ,cell) tuple''' 
		lactor_hiddens = [None]*self.num_slaves*self.num_agents
		lactor_hiddens_forward = [None]*self.num_slaves*self.num_agents
		lactor_terminated = [False]*self.num_slaves*self.num_agents

		for i in range(self.num_slaves):
			for j in range(self.num_agents):
				lactor_states[i*self.num_agents+j] = self.env.getState(i,j)
				lactor_hiddens[i*self.num_agents+j] = self.lactor_model[0].init_hidden(1)
				lactor_hiddens[i*self.num_agents+j] = (lactor_hiddens[i*self.num_agents+j][0].cpu().detach().numpy(), \
							lactor_hiddens[i*self.num_agents+j][1].cpu().detach().numpy())
				lactor_hiddens_forward[i*self.num_agents+j] = self.lactor_model[0].init_hidden(1)

		learningTeam = random.randrange(0,2)
		'''Fixed to team 0'''
		learningTeam = 0
		teamDic = {0: 0, 1: 1}

		local_step = 0
		counter = 0

		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')

			useHardCoded = True


			''' Scheduler Part '''
			for i in range(self.num_slaves):
				scheduler_a_dist_slave = []
				scheduler_v_slave = []
				scheduler_hiddens_slave = []
				for j in range(self.num_agents):
					# if not useHardCoded:
						# if teamDic[j] == learningTeam:
						# 	scheduler_a_dist_slave_agent,scheduler_v_slave_agent, scheduler_hiddens_slave_agent = self.scheduler_model[0].forward(\
						# 		Tensor([scheduler_states[i*self.num_agents+j]]),(Tensor(scheduler_hiddens[i*self.num_agents+j][0]), Tensor(scheduler_hiddens[i*self.num_agents+j][1])), self.env.getNumIterations())
						# else :
						# 	scheduler_a_dist_slave_agent,scheduler_v_slave_agent, scheduler_hiddens_slave_agent = self.scheduler_model[i*self.num_agents+1].forward(\
						# 		Tensor([scheduler_states[i*self.num_agents+j]]),(Tensor(scheduler_hiddens[i*self.num_agents+j][0]), Tensor(scheduler_hiddens[i*self.num_agents+j][1])), self.env.getNumIterations())
						# scheduler_a_dist_slave.append(scheduler_a_dist_slave_agent)
						# scheduler_v_slave.append(scheduler_v_slave_agent)
						# scheduler_hiddens_slave.append((scheduler_hiddens_slave_agent[0].cpu().detach().numpy(), scheduler_hiddens_slave_agent[1].cpu().detach().numpy()))
						# scheduler_actions[i*self.num_agents+j] = scheduler_a_dist_slave[j].sample().cpu().detach().numpy()[0][0];		

					if useHardCoded :
						if teamDic[j] == learningTeam:
							scheduler_a_dist_slave_agent,scheduler_v_slave_agent, scheduler_hiddens_slave_agent = self.scheduler_model[0].forward(\
								Tensor([scheduler_states[i*self.num_agents+j]]),(Tensor(scheduler_hiddens[i*self.num_agents+j][0]), Tensor(scheduler_hiddens[i*self.num_agents+j][1])))
							scheduler_a_dist_slave.append(scheduler_a_dist_slave_agent)
							scheduler_v_slave.append(scheduler_v_slave_agent)
							
							scheduler_hiddens_slave.append((scheduler_hiddens_slave_agent[0].cpu().detach().numpy(), scheduler_hiddens_slave_agent[1].cpu().detach().numpy()))

							explorationFlag = random.randrange(0,3)
							if explorationFlag == 0:
								scheduler_actions[i*self.num_agents+j] = scheduler_a_dist_slave[j].sample().cpu().detach().numpy()[0][0];
							else :
								scheduler_actions[i*self.num_agents+j] = scheduler_a_dist_slave[j].loc.cpu().detach().numpy()[0][0];

						else :
							'''dummy'''
							# scheduler_a_dist_slave_agent,scheduler_v_slave_agent, scheduler_hiddens_slave_agent = self.model[0].forward(\
							# 	Tensor([scheduler_states[i*self.num_agents+j]]),(Tensor(scheduler_hiddens[i*self.num_agents+j][0]), Tensor(scheduler_hiddens[i*self.num_agents+j][1])), self.env.getNumIterations())
							# scheduler_a_dist_slave.append(scheduler_a_dist_slave_agent)
							# scheduler_v_slave.append(scheduler_v_slave_agent)
							# scheduler_hiddens_slave.append((scheduler_hiddens_slave_agent[0].cpu().detach().numpy(), scheduler_hiddens_slave_agent[1].cpu().detach().numpy()))
							scheduler_actions[i*self.num_agents+j] = self.getHardcodedAction(i, j);

				for j in range(self.num_agents):
					if teamDic[j] == learningTeam:
						scheduler_logprobs[i*self.num_agents+j] = scheduler_a_dist_slave[j].log_prob(Tensor(scheduler_actions[i*self.num_agents+j]))\
							.cpu().detach().numpy().reshape(-1)[0];
						scheduler_values[i*self.num_agents+j] = scheduler_v_slave[j].cpu().detach().numpy().reshape(-1)[0];
						scheduler_hiddens_forward[i*self.num_agents+j] = scheduler_hiddens_slave[j]


				''' Set the Linear Actor state with scheduler action '''
				for j in range(self.num_agents):
					# if teamDic[j] == learningTeam:

						# self.env.setLinearActorState(i, j, scheduler_actions[i*self.num_agents+j])
						# lactor_states[i*self.num_agents+j] = self.env.getLinearActorState(i,j)
					self.env.setAction(scheduler_actions[i*self.num_agents+j], i, j);
					# else:



			''' Linear Actor Part '''
			# for i in range(self.num_slaves):
			# 	lactor_a_dist_slave = []
			# 	lactor_v_slave = []
			# 	lactor_hiddens_slave = []
			# 	for j in range(self.num_agents):
			# 		# if not useHardCoded:
			# 		# 	if teamDic[j] == learningTeam:
			# 		# 		lactor_a_dist_slave_agent,lactor_v_slave_agent, lactor_hiddens_slave_agent = self.lactor_model[0].forward(\
			# 		# 			Tensor([lactor_states[i*self.num_agents+j]]),(Tensor(lactor_hiddens[i*self.num_agents+j][0]), Tensor(lactor_hiddens[i*self.num_agents+j][1])), self.env.getNumIterations())
			# 		# 	else :
			# 		# 		lactor_a_dist_slave_agent,lactor_v_slave_agent, lactor_hiddens_slave_agent = self.lactor_model[i*self.num_agents+1].forward(\
			# 		# 			Tensor([lactor_states[i*self.num_agents+j]]),(Tensor(lactor_hiddens[i*self.num_agents+j][0]), Tensor(lactor_hiddens[i*self.num_agents+j][1])), self.env.getNumIterations())
			# 		# 	lactor_a_dist_slave.append(lactor_a_dist_slave_agent)
			# 		# 	lactor_v_slave.append(lactor_v_slave_agent)
			# 		# 	lactor_hiddens_slave.append((lactor_hiddens_slave_agent[0].cpu().detach().numpy(), lactor_hiddens_slave_agent[1].cpu().detach().numpy()))
			# 		# 	lactor_actions[i*self.num_agents+j] = lactor_a_dist_slave[j].sample().cpu().detach().numpy()[0][0];		

			# 		if useHardCoded :
			# 			if teamDic[j] == learningTeam:
			# 				lactor_a_dist_slave_agent,lactor_v_slave_agent, lactor_hiddens_slave_agent = self.lactor_model[0].forward(\
			# 					Tensor([lactor_states[i*self.num_agents+j]]),(Tensor(lactor_hiddens[i*self.num_agents+j][0]), Tensor(lactor_hiddens[i*self.num_agents+j][1])))
			# 				lactor_a_dist_slave.append(lactor_a_dist_slave_agent)
			# 				lactor_v_slave.append(lactor_v_slave_agent)
			# 				lactor_hiddens_slave.append((lactor_hiddens_slave_agent[0].cpu().detach().numpy(), lactor_hiddens_slave_agent[1].cpu().detach().numpy()))
			# 				lactor_actions[i*self.num_agents+j] = lactor_a_dist_slave[j].sample().cpu().detach().numpy()[0][0];		
			# 			else :
			# 				lactor_actions[i*self.num_agents+j] = self.getHardcodedAction(i, j);

			# 	for j in range(self.num_agents):
			# 		if teamDic[j] == learningTeam:
			# 			lactor_logprobs[i*self.num_agents+j] = lactor_a_dist_slave[j].log_prob(Tensor(lactor_actions[i*self.num_agents+j]))\
			# 				.cpu().detach().numpy().reshape(-1)[0];
			# 			lactor_values[i*self.num_agents+j] = lactor_v_slave[j].cpu().detach().numpy().reshape(-1)[0];
			# 			lactor_hiddens_forward[i*self.num_agents+j] = lactor_hiddens_slave[j]
				
			# 	for j in range(self.num_agents):
			# 		# print(lactor_actions[i*self.num_agents+j])
			# 		self.env.setAction(lactor_actions[i*self.num_agents+j], i, j);						


			self.env.stepsAtOnce()


			for i in range(self.num_slaves):
				nan_occur = False
				terminated_state = True
				for k in range(self.num_agents):
					if teamDic[k] == learningTeam:
						# lactor_rewards[i*self.num_agents+k] = self.env.getLinearActorReward(i, k)
						scheduler_rewards[i*self.num_agents+k] = self.env.getSchedulerReward(i, k) #+ lactor_rewards[i*self.num_agents+k]
						# print(	scheduler_rewards[i*self.num_agents+k])
						if np.any(np.isnan(scheduler_rewards[i*self.num_agents+k])):# or np.any(np.isnan(lactor_rewards[i*self.num_agents+k])):
							nan_occur = True
						if np.any(np.isnan(scheduler_states[i*self.num_agents+k])) or np.any(np.isnan(scheduler_actions[i*self.num_agents+k])):
							# or np.any(np.isnan(lactor_states[i*self.num_agents+k])) or np.any(np.isnan(lactor_actions[i*self.num_agents+k])):
							nan_occur = True

				if nan_occur is True:
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam:
							self.total_scheduler_episodes.append(self.scheduler_episodes[i][k])
							self.scheduler_episodes[i][k] = RNNEpisodeBuffer()

							# self.total_lactor_episodes.append(self.lactor_episodes[i][k])
							# self.lactor_episodes[i][k] = RNNEpisodeBuffer()
					self.env.reset(i)


				if self.env.isTerminalState(i) is False:
					terminated_state = False
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam:
							self.scheduler_episodes[i][k].push(scheduler_states[i*self.num_agents+k], scheduler_actions[i*self.num_agents+k],\
								scheduler_rewards[i*self.num_agents+k], scheduler_values[i*self.num_agents+k], scheduler_logprobs[i*self.num_agents+k], scheduler_hiddens[i*self.num_agents+k])
							# self.lactor_episodes[i][k].push(lactor_states[i*self.num_agents+k], lactor_actions[i*self.num_agents+k],\
							# 	lactor_rewards[i*self.num_agents+k], lactor_values[i*self.num_agents+k], lactor_logprobs[i*self.num_agents+k], lactor_hiddens[i*self.num_agents+k])

							local_step += 1

				if terminated_state is True:
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam:
							self.scheduler_episodes[i][k].push(scheduler_states[i*self.num_agents+k], scheduler_actions[i*self.num_agents+k],\
								scheduler_rewards[i*self.num_agents+k], scheduler_values[i*self.num_agents+k], scheduler_logprobs[i*self.num_agents+k], scheduler_hiddens[i*self.num_agents+k])
							self.total_scheduler_episodes.append(self.scheduler_episodes[i][k])
							self.scheduler_episodes[i][k] = RNNEpisodeBuffer()

							# self.lactor_episodes[i][k].push(lactor_states[i*self.num_agents+k], lactor_actions[i*self.num_agents+k],\
							# 	lactor_rewards[i*self.num_agents+k], lactor_values[i*self.num_agents+k], lactor_logprobs[i*self.num_agents+k], lactor_hiddens[i*self.num_agents+k])
							# self.total_lactor_episodes.append(self.lactor_episodes[i][k])
							# self.lactor_episodes[i][k] = RNNEpisodeBuffer()
					self.env.reset(i)

			if local_step >= self.buffer_size:
				for i in range(self.num_slaves):
					self.total_scheduler_episodes.append(self.scheduler_episodes[i][k])
					self.scheduler_episodes[i][k] = RNNEpisodeBuffer()

					# self.total_lactor_episodes.append(self.lactor_episodes[i][k])
					# self.lactor_episodes[i][k] = RNNEpisodeBuffer()
				self.env.reset(i)
				break

			# get the updated state and updated hidden
			for i in range(self.num_slaves):
				for j in range(self.num_agents):
					scheduler_states[i*self.num_agents+j] = self.env.getSchedulerState(i,j)
					scheduler_hiddens[i*self.num_agents+j] = scheduler_hiddens_forward[i*self.num_agents+j]
					# lactor_states[i*self.num_agents+j] = self.env.getState(i,j)
					# lactor_hiddens[i*self.num_agents+j] = lactor_hiddens_forward[i*self.num_agents+j]

		self.env.endOfIteration()
		print('SIM : {}'.format(local_step))

	def generateHindsightTransitions(self):
		print('Hindsight processing --', end=" ")
		self.total_scheduler_hindsight_episodes = []
		for episode_index in range(len(self.total_scheduler_episodes)):
			scheduler_episode = self.total_scheduler_episodes[episode_index];
			lactor_episode = self.total_lactor_episodes[episode_index];
			data = scheduler_episode.getData()
			lactor_data = lactor_episode.getData()
			len_episode = len(data)
			if len_episode == 0:
				continue
			hindsight_goal_per_episode = 4
			random_segment_point = []
			for i in range(hindsight_goal_per_episode):
				rand_num = random.randrange(0, len_episode)
				if (rand_num in random_segment_point) is False:
					random_segment_point.append(rand_num)
			random_segment_point.sort()

			index_count = 0
			self.env.setHindsightGoal(data[random_segment_point[index_count]].s, data[random_segment_point[index_count]].a)

			scheduler_hindsight_episodes =RNNEpisodeBuffer()
			for timeStep in range(len_episode):
				# if timeStep <= random_segment_point[index_count]:
				cur_state = data[timeStep].s
				cur_action = data[timeStep].a
				cur_hidden = data[timeStep].hidden
				cur_logprob = data[timeStep].logprob
				hindsight_state = self.env.getHindsightState(cur_state)
				hindsight_reward = self.env.getHindsightReward(cur_state, cur_action)

				hindsight_lactor_reward = lactor_data[timeStep].r

				hindsight_reward += hindsight_lactor_reward

				# print(Tensor(hindsight_state).size())
				_,hindsight_v_slave_agent, hindsight_hidden= self.scheduler_model[0].forward(\
						Tensor([hindsight_state]), (Tensor(cur_hidden[0]), Tensor(cur_hidden[1])))

				hindsight_value = hindsight_v_slave_agent.cpu().detach().numpy().reshape(-1)[0];

				scheduler_hindsight_episodes.push(hindsight_state, cur_action,\
						hindsight_reward, hindsight_value, cur_logprob, (hindsight_hidden[0].cpu().detach().numpy(), hindsight_hidden[1].cpu().detach().numpy()))

				if timeStep == random_segment_point[index_count]:
					index_count += 1
					if index_count < len(random_segment_point):
						# print(index_count)
						self.env.setHindsightGoal(data[random_segment_point[index_count]].s, data[random_segment_point[index_count]].a)
					else:
						break;
			self.total_scheduler_hindsight_episodes.append(scheduler_hindsight_episodes)

		# print(len(self.total_scheduler_episodes))

		self.total_scheduler_episodes = self.total_scheduler_episodes + self.total_scheduler_hindsight_episodes

		# print(len(self.total_scheduler_episodes))
			# print(random_segment_point)
		print('Complete')



	def optimizeSchedulerNN(self):
		all_rnn_replay_buffer= np.array(self.scheduler_buffer.buffer)
		for j in range(self.num_epochs):
			# Get truncated transitions. The transitions will be splited by size self.trunc_size
			all_segmented_transitions = []
			for rnn_replay_buffer in all_rnn_replay_buffer:
				rnn_replay_buffer_size = len(rnn_replay_buffer.buffer)

				# We will fill the remainder with 0.
				for i in range(rnn_replay_buffer_size//self.trunc_size):
					segmented_transitions = np.array(rnn_replay_buffer.buffer)[i*self.trunc_size:(i+1)*self.trunc_size]
					all_segmented_transitions.append(segmented_transitions)
					# zero padding
					if (i+2)*self.trunc_size > rnn_replay_buffer_size :
						segmented_transitions = [RNNTransition(None,None,None,None,None,None,None) for x in range(self.trunc_size)]
						segmented_transitions[:rnn_replay_buffer_size - (i+1)*self.trunc_size] = \
							np.array(rnn_replay_buffer.buffer)[(i+1)*self.trunc_size:rnn_replay_buffer_size]
						all_segmented_transitions.append(segmented_transitions)

			# Shuffle the segmented transition (order of episodes)
			np.random.shuffle(all_segmented_transitions)


			# We will get loss with trunc_size in a batch. And then we use the mean value.
			for i in range(len(all_segmented_transitions)//self.batch_size):
				batch_segmented_transitions = all_segmented_transitions[i*self.batch_size:(i+1)*self.batch_size]

				non_None_batch_list = [[] for x in range(self.trunc_size)]


				loss_list = [Tensor([0])] * self.trunc_size
				hidden_list = [None]*self.trunc_size

				a_dist_list = [None]*self.trunc_size
				v_list = [None]*self.trunc_size
				loss = Tensor(torch.zeros(1).cuda())

				for timeStep in range(self.trunc_size):

					non_None_batch_segmented_transitions = []

					# Make non-None list of batch_segmented_transitions
					if timeStep == 0:
						for k in range(self.batch_size):
							if RNNTransition(*batch_segmented_transitions[k][timeStep]).s is not None :
								non_None_batch_segmented_transitions.append(batch_segmented_transitions[k][timeStep])
								non_None_batch_list[timeStep].append(k)
					else :
						for k in non_None_batch_list[timeStep-1]:
							if RNNTransition(*batch_segmented_transitions[k][timeStep]).s is not None :
								non_None_batch_segmented_transitions.append(batch_segmented_transitions[k][timeStep])
								non_None_batch_list[timeStep].append(k)

					batch = RNNTransition(*zip(*non_None_batch_segmented_transitions))

					stack_s = np.vstack(batch.s).astype(np.float32)
					stack_a = np.vstack(batch.a).astype(np.float32)
					stack_lp = np.vstack(batch.logprob).astype(np.float32)
					stack_td = np.vstack(batch.TD).astype(np.float32)
					stack_gae = np.vstack(batch.GAE).astype(np.float32)
					stack_hidden_h = np.vstack(batch.hidden_h).astype(np.float32)
					stack_hidden_c = np.vstack(batch.hidden_c).astype(np.float32)


					num_layers =self.scheduler_model[0].num_layers
					stack_hidden = [Tensor(np.reshape(stack_hidden_h, (num_layers,len(non_None_batch_list[timeStep]),-1))), 
										Tensor(np.reshape(stack_hidden_c, (num_layers,len(non_None_batch_list[timeStep]),-1)))]
					# stack_hidden = [None,None]


					if timeStep >= 1:
						batch_count = 0
						for k in non_None_batch_list[timeStep]:
							for l in range(len(non_None_batch_list[timeStep-1])) :
								if non_None_batch_list[timeStep-1][l] == k:
									for t in range(num_layers):
										stack_hidden[0][t][batch_count] = hidden_list[timeStep-1][0][t][l]
										stack_hidden[1][t][batch_count] = hidden_list[timeStep-1][1][t][l]
									batch_count += 1


					# stack_hidden = list(self.model[0].init_hidden(len(non_None_batch_list[timeStep])))



					if timeStep % 16 == 0:
						stack_hidden[0] = stack_hidden[0].detach()
						stack_hidden[1] = stack_hidden[1].detach()
						loss = Tensor(torch.zeros(1).cuda())

					a_dist,v,cur_stack_hidden = self.scheduler_model[0].forward(Tensor(stack_s), stack_hidden)	

					hidden_list[timeStep] = list(cur_stack_hidden)

					# print(timeStep)
					if timeStep >= self.burn_in_size :
						loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()

						'''Actor Loss'''
						ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
						stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+1E-5)
						stack_gae = Tensor(stack_gae)
						surrogate1 = ratio * stack_gae
						surrogate2 = torch.clamp(ratio, min=1.0-self.clip_ratio, max=1.0+self.clip_ratio) * stack_gae
						# print(ratio.mean(), end=" ")
						# print(surrogate1.mean(), end= " ")
						# print(surrogate2.mean())
						# print(surrogate1.mean(), end=" ")
						# print(surrogate2.mean())
						loss_actor = - torch.min(surrogate1, surrogate2).mean()
						# loss_actor = - surrogate2.mean()

						'''Entropy Loss'''
						loss_entropy = - self.w_entropy * a_dist.entropy().mean()

						self.scheduler_loss_actor = loss_actor.cpu().detach().numpy().tolist()
						self.scheduler_loss_critic = loss_critic.cpu().detach().numpy().tolist()

						loss += loss_actor + loss_entropy + loss_critic

						if timeStep % 16 == 15:
							self.scheduler_optimizer.zero_grad()

							# start = time.time()
							loss.backward(retain_graph=True)
							# print("time :", time.time() - start)

							for param in self.scheduler_model[0].parameters():
								if param.grad is not None:
									param.grad.data.clamp_(-0.5, 0.5)
							self.scheduler_optimizer.step()
						elif timeStep == (self.trunc_size-1):
							self.scheduler_optimizer.zero_grad()

							# start = time.time()
							loss.backward(retain_graph=True)
							# print("time :", time.time() - start)

							for param in self.scheduler_model[0].parameters():
								if param.grad is not None:
									param.grad.data.clamp_(-0.5, 0.5)
							self.scheduler_optimizer.step()

			print('Optimizing scheduler nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')


	def optimizeSchedulerNN1(self):
		all_rnn_replay_buffer= np.array(self.scheduler_buffer.buffer)
		for j in range(self.num_epochs):
			# Get truncated transitions. The transitions will be splited by size self.trunc_size
			all_segmented_transitions = []
			for rnn_replay_buffer in all_rnn_replay_buffer:
				rnn_replay_buffer_size = len(rnn_replay_buffer.buffer)

				# We will fill the remainder with 0.
				# for i in range(rnn_replay_buffer_size//self.trunc_size):
				# 	segmented_transitions = np.array(rnn_replay_buffer.buffer)[i*self.trunc_size:(i+1)*self.trunc_size]
				# 	all_segmented_transitions.append(segmented_transitions)
				# 	# zero padding
				# 	if (i+2)*self.trunc_size > rnn_replay_buffer_size :
				# 		segmented_transitions = [RNNTransition(None,None,None,None,None,None,None) for x in range(self.trunc_size)]
				# 		segmented_transitions[:rnn_replay_buffer_size - (i+1)*self.trunc_size] = \
				# 			np.array(rnn_replay_buffer.buffer)[(i+1)*self.trunc_size:rnn_replay_buffer_size]
				# 		all_segmented_transitions.append(segmented_transitions)

				for i in range(rnn_replay_buffer_size):
					all_segmented_transitions.append(rnn_replay_buffer.buffer[i])

			# print(len(all_segmented_transitions))
			# Shuffle the segmented transition (order of episodes)
			np.random.shuffle(all_segmented_transitions)

			for i in range(len(all_segmented_transitions)//self.batch_size):
				batch_segmented_transitions = all_segmented_transitions[i*self.batch_size:(i+1)*self.batch_size]

				batch = RNNTransition(*zip(*batch_segmented_transitions))

				stack_s = np.vstack(batch.s).astype(np.float32)
				stack_a = np.vstack(batch.a).astype(np.float32)
				stack_lp = np.vstack(batch.logprob).astype(np.float32)
				stack_td = np.vstack(batch.TD).astype(np.float32)
				stack_gae = np.vstack(batch.GAE).astype(np.float32)
				stack_hidden_h = np.vstack(batch.hidden_h).astype(np.float32)
				stack_hidden_c = np.vstack(batch.hidden_c).astype(np.float32)

				num_layers =self.scheduler_model[0].num_layers
				stack_hidden = [Tensor(np.reshape(stack_hidden_h, (num_layers,self.batch_size,-1))), 
									Tensor(np.reshape(stack_hidden_c, (num_layers,self.batch_size,-1)))]
				# stack_hidden = [None,None]


				# stack_hidden = list(self.model[0].init_hidden(len(non_None_batch_list[timeStep])))
				stack_hidden[0] = stack_hidden[0].detach()
				stack_hidden[1] = stack_hidden[1].detach()


				a_dist,v,cur_stack_hidden = self.scheduler_model[0].forward(Tensor(stack_s), stack_hidden)	


				loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()

				'''Actor Loss'''
				ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
				stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+1E-5)
				stack_gae = Tensor(stack_gae)
				surrogate1 = ratio * stack_gae
				surrogate2 = torch.clamp(ratio, min=1.0-self.clip_ratio, max=1.0+self.clip_ratio) * stack_gae

				loss_actor = - torch.min(surrogate1, surrogate2).mean()
				# loss_actor = - surrogate2.mean()

				'''Entropy Loss'''
				loss_entropy = - self.w_entropy * a_dist.entropy().mean()

				self.scheduler_loss_actor = loss_actor.cpu().detach().numpy().tolist()
				self.scheduler_loss_critic = loss_critic.cpu().detach().numpy().tolist()

				loss = loss_actor + loss_entropy + loss_critic
				self.scheduler_optimizer.zero_grad()

				# start = time.time()
				loss.backward(retain_graph=True)
				# print("time :", time.time() - start)

				for param in self.scheduler_model[0].parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5, 0.5)
				self.scheduler_optimizer.step()

			print('Optimizing scheduler nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')

	def optimizeLActorNN(self):
		all_rnn_replay_buffer= np.array(self.lactor_buffer.buffer)
		for j in range(self.num_epochs):
			# Get truncated transitions. The transitions will be splited by size self.trunc_size
			all_segmented_transitions = []
			for rnn_replay_buffer in all_rnn_replay_buffer:
				rnn_replay_buffer_size = len(rnn_replay_buffer.buffer)

				# We will fill the remainder with 0.
				for i in range(rnn_replay_buffer_size//self.trunc_size):
					segmented_transitions = np.array(rnn_replay_buffer.buffer)[i*self.trunc_size:(i+1)*self.trunc_size]
					all_segmented_transitions.append(segmented_transitions)
					# zero padding
					if (i+2)*self.trunc_size > rnn_replay_buffer_size :
						segmented_transitions = [RNNTransition(None,None,None,None,None,None,None) for x in range(self.trunc_size)]
						segmented_transitions[:rnn_replay_buffer_size - (i+1)*self.trunc_size] = \
							np.array(rnn_replay_buffer.buffer)[(i+1)*self.trunc_size:rnn_replay_buffer_size]
						all_segmented_transitions.append(segmented_transitions)

			# Shuffle the segmented transition (order of episodes)
			np.random.shuffle(all_segmented_transitions)


			# We will get loss with trunc_size in a batch. And then we use the mean value.
			for i in range(len(all_segmented_transitions)//self.batch_size):
				batch_segmented_transitions = all_segmented_transitions[i*self.batch_size:(i+1)*self.batch_size]

				non_None_batch_list = [[] for x in range(self.trunc_size)]


				loss_list = [Tensor([0])] * self.trunc_size
				hidden_list = [None]*self.trunc_size

				a_dist_list = [None]*self.trunc_size
				v_list = [None]*self.trunc_size
				loss = Tensor(torch.zeros(1).cuda())

				for timeStep in range(self.trunc_size):

					non_None_batch_segmented_transitions = []

					# Make non-None list of batch_segmented_transitions
					if timeStep == 0:
						for k in range(self.batch_size):
							if RNNTransition(*batch_segmented_transitions[k][timeStep]).s is not None :
								non_None_batch_segmented_transitions.append(batch_segmented_transitions[k][timeStep])
								non_None_batch_list[timeStep].append(k)
					else :
						for k in non_None_batch_list[timeStep-1]:
							if RNNTransition(*batch_segmented_transitions[k][timeStep]).s is not None :
								non_None_batch_segmented_transitions.append(batch_segmented_transitions[k][timeStep])
								non_None_batch_list[timeStep].append(k)

					batch = RNNTransition(*zip(*non_None_batch_segmented_transitions))

					stack_s = np.vstack(batch.s).astype(np.float32)
					stack_a = np.vstack(batch.a).astype(np.float32)
					stack_lp = np.vstack(batch.logprob).astype(np.float32)
					stack_td = np.vstack(batch.TD).astype(np.float32)
					stack_gae = np.vstack(batch.GAE).astype(np.float32)
					stack_hidden_h = np.vstack(batch.hidden_h).astype(np.float32)
					stack_hidden_c = np.vstack(batch.hidden_c).astype(np.float32)


					num_layers =self.lactor_model[0].num_layers
					stack_hidden = [Tensor(np.reshape(stack_hidden_h, (num_layers,len(non_None_batch_list[timeStep]),-1))), 
										Tensor(np.reshape(stack_hidden_c, (num_layers,len(non_None_batch_list[timeStep]),-1)))]
					# stack_hidden = [None,None]


					if timeStep >= 1:
						batch_count = 0
						for k in non_None_batch_list[timeStep]:
							for l in range(len(non_None_batch_list[timeStep-1])) :
								if non_None_batch_list[timeStep-1][l] == k:
									for t in range(num_layers):
										stack_hidden[0][t][batch_count] = hidden_list[timeStep-1][0][t][l]
										stack_hidden[1][t][batch_count] = hidden_list[timeStep-1][1][t][l]
									batch_count += 1


					# stack_hidden = list(self.model[0].init_hidden(len(non_None_batch_list[timeStep])))



					if timeStep % 8 == 0:
						stack_hidden[0] = stack_hidden[0].detach()
						stack_hidden[1] = stack_hidden[1].detach()
						loss = Tensor(torch.zeros(1).cuda())

					a_dist,v,cur_stack_hidden = self.lactor_model[0].forward(Tensor(stack_s), stack_hidden)	

					hidden_list[timeStep] = list(cur_stack_hidden)


					if timeStep >= self.burn_in_size :
						loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()

						'''Actor Loss'''
						ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
						stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+1E-5)
						stack_gae = Tensor(stack_gae)
						surrogate1 = ratio * stack_gae
						surrogate2 = torch.clamp(ratio, min=1.0-self.clip_ratio, max=1.0+self.clip_ratio) * stack_gae
						# print(ratio.mean(), end=" ")
						# print(surrogate1.mean(), end= " ")
						# print(surrogate2.mean())
						# print(surrogate1.mean(), end=" ")
						# print(surrogate2.mean())
						loss_actor = - torch.min(surrogate1, surrogate2).mean()
						# loss_actor = - surrogate2.mean()

						'''Entropy Loss'''
						loss_entropy = - self.w_entropy * a_dist.entropy().mean()

						self.lactor_loss_actor = loss_actor.cpu().detach().numpy().tolist()
						self.lactor_loss_critic = loss_critic.cpu().detach().numpy().tolist()

						loss += loss_actor + loss_entropy + loss_critic

						if timeStep % 8 == 7:
							self.lactor_optimizer.zero_grad()

							# start = time.time()
							loss.backward(retain_graph=True)
							# print("time :", time.time() - start)

							for param in self.lactor_model[0].parameters():
								if param.grad is not None:
									param.grad.data.clamp_(-0.5, 0.5)
							self.lactor_optimizer.step()
			print('Optimizing LActor nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')


	def optimizeModel(self):
		self.computeTDandGAE()
		self.optimizeSchedulerNN()
		# self.optimizeLActorNN()


	def train(self):
		frac = 1.0
		self.learning_rate = self.default_learning_rate*frac
		self.clip_ratio = self.default_clip_ratio*frac
		for param_group in self.scheduler_optimizer.param_groups:
			param_group['lr'] = self.learning_rate
		for param_group in self.lactor_optimizer.param_groups:
			param_group['lr'] = self.learning_rate
		self.generateTransitions();
		# self.generateHindsightTransitions();
		self.optimizeModel()
	def loadModel(self,path,index):
		self.scheduler_model[index].load('../nn/'+path+"_sc.pt")
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
		if self.scheduler_max_return < self.scheduler_sum_return/self.num_episode:
			self.scheduler_max_return = self.scheduler_sum_return/self.num_episode
			self.scheduler_max_return_epoch = self.num_evaluation
		if self.lactor_max_return < self.lactor_sum_return/self.num_episode:
			self.lactor_max_return = self.lactor_sum_return/self.num_episode
			self.lactor_max_return_epoch = self.num_evaluation

		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,h,m,s))
		print('||--------------SchedulerNN------------------')
		print('||Loss Actor               : {:.4f}'.format(self.scheduler_loss_actor))
		print('||Loss Critic              : {:.4f}'.format(self.scheduler_loss_critic))
		print('||Noise                    : {:.3f}'.format(self.scheduler_model[0].log_std.exp().mean()))		
		print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
		print('||Num Transition           : {}'.format(self.num_tuple))
		print('||Num Episode              : {}'.format(self.num_episode))
		print('||Avg Return per episode   : {:.3f}'.format(self.scheduler_sum_return/self.num_episode))
		print('||Avg Reward per transition: {:.3f}'.format(self.scheduler_sum_return/self.num_tuple))
		print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
		print('||Max Avg Retun So far     : {:.3f} at #{}'.format(self.scheduler_max_return,self.scheduler_max_return_epoch))
		# print('||-----------------LActorNN------------------')
		# print('||Loss Actor               : {:.4f}'.format(self.lactor_loss_actor))
		# print('||Loss Critic              : {:.4f}'.format(self.lactor_loss_critic))
		# print('||Noise                    : {:.3f}'.format(self.lactor_model[0].log_std.exp().mean()))		
		# print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
		# print('||Num Transition           : {}'.format(self.num_tuple))
		# print('||Num Episode              : {}'.format(self.num_episode))
		# print('||Avg Return per episode   : {:.3f}'.format(self.lactor_sum_return/self.num_episode))
		# print('||Avg Reward per transition: {:.3f}'.format(self.lactor_sum_return/self.num_tuple))
		# print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
		# print('||Max Avg Retun So far     : {:.3f} at #{}'.format(self.lactor_max_return,self.lactor_max_return_epoch))
		self.scheduler_rewards.append(self.scheduler_sum_return/self.num_episode)
		# self.lactor_rewards.append(self.lactor_sum_return/self.num_episode)
		
		self.saveModel()
		
		print('=============================================')
		return np.array(self.scheduler_rewards)#, np.array(self.lactor_rewards)


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
	slac = SLAC()
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')
	parser.add_argument('-iter','--iteration',help='num iterations')
	parser.add_argument('-n','--name',help='name of training setting')

	args =parser.parse_args()
	
	graph_name = ''
	
	if args.model is not None:
		slac.loadModel(args.model, 0)
		if args.iteration is not None:
			slac.num_evaluation = int(args.iteration)
			for i in range(int(args.iteration)):
				slac.env.endOfIteration()
	if args.name is not None:
		graph_name = args.name

	else:
		slac.saveModel()
	print('num states: {}, num actions: {}'.format(slac.env.getNumState(),slac.env.getNumAction()))
	# for i in range(ppo.max_iteration-5):
	for i in range(5000000):
		slac.train()
		# scheduler_rewards, lactor_rewards = slac.evaluate()
		scheduler_rewards = slac.evaluate()
		plot(scheduler_rewards, graph_name + ' scheduler reward',0,False)

		# plot(lactor_rewards,'lactor reward',1,False)
