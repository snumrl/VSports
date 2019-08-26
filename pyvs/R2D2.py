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
Episode = namedtuple('Episode', ('s','a','r','value','logprob'))
RNNEpisode = namedtuple('RNNEpisode', ('s','a','r','value','logprob','hidden'))
# RNNEpisode = namedtuple('RNNEpisode', ('s','a','r','value','logprob','hidden_h','hidden_c'))

class EpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def push(self, *args):
		self.data.append(Episode(*args))

	def getData(self):
		return self.data

class RNNEpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def push(self, *args):
		self.data.append(RNNEpisode(*args))

	def getData(self):
		return self.data

Transition = namedtuple('Transition',('s','a','logprob','TD','GAE'))

RNNTransition = namedtuple('RNNTransition',('s','a','logprob','TD','GAE','hidden_h','hidden_c'))

class RNNReplayBuffer(object):
	def __init__(self, buff_size = 10000):
		super(RNNReplayBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def push(self,*args):
		self.buffer.append(RNNTransition(*args))

	def clear(self):
		self.buffer.clear()


class ReplayBuffer(object):
	def __init__(self, buff_size = 10000):
		super(ReplayBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def push(self,*args):
		self.buffer.append(Transition(*args))

	def clear(self):
		self.buffer.clear()

class RNNBuffer(object):
	def __init__(self, buff_size = 10000):
		super(RNNBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def push(self,replay_buffer):
		self.buffer.append(replay_buffer)

	def clear(self):
		self.buffer.clear()

class PPO(object):
	def __init__(self):
		np.random.seed(seed = int(time.time()))
		self.env = Env(600)
		self.num_slaves = 16
		self.num_agents = 4
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

		self.buffer_size = 4*2048
		self.batch_size = 128
		self.trunc_size = 64
		self.burn_in_size = 32
		# self.replay_buffer = ReplayBuffer(30000)
		self.rnn_buffer = RNNBuffer(30000)

		useMap = False;

		self.model = [None]*self.num_slaves*self.num_agents
		for i in range(self.num_slaves*self.num_agents):
			# exit()
			if useMap:
				self.model[i] = CombinedSimulationNN(self.num_state, self.num_action)
			else:
				self.model[i] = NoCNNSimulationNN(self.num_state, self.num_action)

			if use_cuda:
				self.model[i].cuda()

		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio
		self.optimizer = optim.Adam(self.model[0].parameters(), lr=self.learning_rate)
		# print(self.model[0].parameters())
		self.max_iteration = 50000

		self.w_entropy = 0.0001

		self.loss_actor = 0.0
		self.loss_critic = 0.0
		self.rewards = []
		self.sum_return = 0.0
		self.max_return = 0.0
		self.max_return_epoch = 1
		self.tic = time.time()

		# self.episodes = [[EpisodeBuffer() for y in range(self.num_agents)] for x in range(self.num_slaves)]
		self.rnn_episodes = [[RNNEpisodeBuffer() for y in range(self.num_agents)] for x in range(self.num_slaves)]

		self.env.resets()

	# def getCombState(self, index):
	# 	combState = []



	def saveModel(self):
		self.model[0].save('../nn/current.pt')

		if self.max_return_epoch == self.num_evaluation:
			self.model[0].save('../nn/max.pt')
		if self.num_evaluation%20 == 0:
			self.model[0].save('../nn/'+str(self.num_evaluation//20)+'.pt')

	def loadModel(self,path,index):
		self.model[index].load('../nn/'+path+'.pt')

	def computeTDandGAE(self):
		# self.replay_buffer.clear()
		self.rnn_buffer.clear()
		self.sum_return = 0.0
		for epi in self.total_episodes:
			data = epi.getData()
			size = len(data)
			# print("Size : ",size)
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
				self.sum_return += epi_return
				TD = values[:size] + advantages

				rnn_replay_buffer = RNNReplayBuffer(4000)
				# print("size of data ",size)
				for i in range(size):
					# print(actions[i])
					# print(hiddens[i])
					# exit(0)
					rnn_replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i], hiddens[i][0], hiddens[i][1])
				self.rnn_buffer.push(rnn_replay_buffer)


		self.num_episode = len(self.total_episodes)
		self.num_tuple = 0
		for rnn_replay_buffer in self.rnn_buffer.buffer:
			# print(len(rnn_replay_buffer.buffer))
			self.num_tuple += len(rnn_replay_buffer.buffer)
		# self.num_tuple = len(self.replay_buffer.buffer)
		self.num_tuple_so_far += self.num_tuple

	def getHardcodedAction(self, slave_index, agent_index):
		state = self.env.getState(slave_index, agent_index)
		# mapOffset = 6400
		action = []
		# print(state[0:10])
		direction = state[4:6]
		direction_norm = 0
		for i in direction:
			direction_norm += i*i

		direction_norm = math.sqrt(direction_norm)


		# maxvel = 1.5 + 1.5* self.env.getNumIterations()/150.0
		# if maxvel > 3.0:
		# 	maxvel = 3.0
		maxvel = 4.0

		action = (maxvel*direction/direction_norm - state[2:4])
		# for i in range(action.size()):
		# 	if action[i] > 0.3:
		# 		action[i] = 0.3
		# action = np.append(action, [random.randrange(0,3) - 1])
		action = np.append(action, [1.0])
		# action = np.append(action, [0])

		return action

	def generateTransitions(self):
		self.total_episodes = []
		states = [None]*self.num_slaves*self.num_agents
		actions = [None]*self.num_slaves*self.num_agents
		rewards = [None]*self.num_slaves*self.num_agents
		logprobs = [None]*self.num_slaves*self.num_agents
		values = [None]*self.num_slaves*self.num_agents
		# hiddens : (hidden ,cell) tuple
		hiddens = [None]*self.num_slaves*self.num_agents
		hiddens_forward = [None]*self.num_slaves*self.num_agents
		# states_next = [None]*self.num_slaves*self.num_agents
		for i in range(self.num_slaves):
			for j in range(self.num_agents):
				states[i*self.num_agents+j] = self.env.getState(i,j)
				hiddens[i*self.num_agents+j] = self.model[0].init_hidden(1)
				hiddens[i*self.num_agents+j] = (hiddens[i*self.num_agents+j][0].cpu().detach().numpy(), \
							hiddens[i*self.num_agents+j][1].cpu().detach().numpy())
				hiddens_forward[i*self.num_agents+j] = self.model[0].init_hidden(1)
		local_step = 0
		terminated = [False]*self.num_slaves*self.num_agents
		counter = 0

		useHardCoded = True


		if not useHardCoded:
			if self.num_evaluation > 20:
				curEvalNum = self.num_evaluation//20

				clipedRandomNormal = 0
				while clipedRandomNormal <= 0.5:
					clipedRandomNormal = np.random.normal(curEvalNum, curEvalNum/2,1)[0]
					if clipedRandomNormal > curEvalNum :
						clipedRandomNormal = 2*curEvalNum - clipedRandomNormal;

				for i in range(self.num_slaves):
					# prevPath = "../nn/"+clipedRandomNormal+".pt";

					self.loadModel(str(int(clipedRandomNormal+0.5)), i*self.num_agents+1)
					# self.loadModel("current", i*self.num_agents+1)
			else :
				for i in range(self.num_slaves):
					self.loadModel("0",i*self.num_agents+1)


		# 0 or 1
		learningTeam = random.randrange(0,2)
		learningTeam = 0
		teamDic = {0: 0, 1: 0, 2: 1, 3:1}

		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')

			# per_agent_state = [[None]*self.num_agents]*self.num_slaves;
			# for i in range(self.num_slaves):
			# 	for j in range(self.num_agents):
			# 		per_agent_state[i][j] = self.env.getState(i,j);

				# a_dist,v = self.model(Tensor(per_agent_state[i]))

				# actions = a_dist.sample().cpu().detach().numpy()

				# logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
				# values = v.cpu().detach().numpy().reshape(-1)

			# logprobs = []
			# values = []

			for i in range(self.num_slaves):
				a_dist_slave = []
				v_slave = []
				actions_slave = []
				logprobs_slave = []
				values_slave = []
				hiddens_slave = []
				for j in range(self.num_agents):
					if not useHardCoded:
						if teamDic[j] == learningTeam or True:
							a_dist_slave_agent,v_slave_agent, hiddens_slave_agent = self.model[0].forward_rnn(\
								Tensor([states[i*self.num_agents+j]]),(Tensor(hiddens[i*self.num_agents+j][0]), Tensor(hiddens[i*self.num_agents+j][1])))
						else :
							a_dist_slave_agent,v_slave_agent, hiddens_slave_agent = self.model[i*self.num_agents+j].forward_rnn(\
								Tensor([states[i*self.num_agents+j]]),(Tensor(hiddens[i*self.num_agents+j][0]), Tensor(hiddens[i*self.num_agents+j][1])))
						a_dist_slave.append(a_dist_slave_agent)
						v_slave.append(v_slave_agent)
						hiddens_slave.append((hiddens_slave_agent[0].cpu().detach().numpy(), hiddens_slave_agent[1].cpu().detach().numpy()))
						actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy()[0][0];		

					else :
						# print(self.model[0](Tensor([states[i*self.num_agents+j]])))
						# exit()
						if teamDic[j] == learningTeam:
							a_dist_slave_agent,v_slave_agent, hiddens_slave_agent = self.model[0].forward_rnn(\
								Tensor([states[i*self.num_agents+j]]),(Tensor(hiddens[i*self.num_agents+j][0]), Tensor(hiddens[i*self.num_agents+j][1])))
							a_dist_slave.append(a_dist_slave_agent)
							v_slave.append(v_slave_agent)
							hiddens_slave.append((hiddens_slave_agent[0].cpu().detach().numpy(), hiddens_slave_agent[1].cpu().detach().numpy()))
							actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy()[0][0];		
						else :
							# dummy
							# a_dist_slave_agent,v_slave_agent = self.model[0](Tensor([states[i*self.num_agents+j]]))
							# a_dist_slave.append(a_dist_slave_agent)
							# v_slave.append(v_slave_agent)
							a_dist_slave_agent,v_slave_agent, hiddens_slave_agent = self.model[0].forward_rnn(\
								Tensor([states[i*self.num_agents+j]]),(Tensor(hiddens[i*self.num_agents+j][0]), Tensor(hiddens[i*self.num_agents+j][1])))
							a_dist_slave.append(a_dist_slave_agent)
							v_slave.append(v_slave_agent)
							hiddens_slave.append((hiddens_slave_agent[0].cpu().detach().numpy(), hiddens_slave_agent[1].cpu().detach().numpy()))
							actions[i*self.num_agents+j] = self.getHardcodedAction(i, j);

				for j in range(self.num_agents):
					if teamDic[j] == learningTeam or True:
						logprobs[i*self.num_agents+j] = a_dist_slave[j].log_prob(Tensor(actions[i*self.num_agents+j]))\
							.cpu().detach().numpy().reshape(-1)[0];
						values[i*self.num_agents+j] = v_slave[j].cpu().detach().numpy().reshape(-1)[0];
						hiddens_forward[i*self.num_agents+j] = hiddens_slave[j]

				for j in range(self.num_agents):
					self.env.setAction(actions[i*self.num_agents+j], i, j);


			# print(values[0])
			# print(rewards[0])

			# print(1111111)
			self.env.stepsAtOnce()
			# self.env.step(0)
			# print(22222)

			for i in range(self.num_slaves):
				nan_occur = False
				terminated_state = True
				for k in range(self.num_agents):
					if teamDic[k] == learningTeam or True:
						rewards[i*self.num_agents+k] = self.env.getReward(i, k)
						if np.any(np.isnan(rewards[i*self.num_agents+k])):
							nan_occur = True
					if np.any(np.isnan(states[i*self.num_agents+k])) or np.any(np.isnan(actions[i*self.num_agents+k])):
						nan_occur = True
										 
					# if i == 2:
					# 	print(states[i*self.num_agents+k][0:10])
					# 	print(actions[i*self.num_agents+k])

				if nan_occur is True:
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or False:
							self.total_episodes.append(self.rnn_episodes[i][k])
							self.rnn_episodes[i][k] = RNNEpisodeBuffer()
					self.env.reset(i)

				if self.env.isTerminalState(i) is False:
					terminated_state = False
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or False:
							# rewards[i*self.num_agents+k] = self.env.getReward(i, k)
							# print(str(i)+" "+str(k), end=' ')
							# print(len(self.episodes[i][k].getData()), end = ' ')
							self.rnn_episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k],\
								rewards[i*self.num_agents+k], values[i*self.num_agents+k], logprobs[i*self.num_agents+k], hiddens[i*self.num_agents+k])
							# print(local_step)
							local_step += 1

				# print(id(self.episodes[0][1].getData()))
				# print(id(self.episodes[1][0].getData()))
				# print(id(self.episodes[2][0].getData()))
				# print(id(self.episodes[3][0].getData()))
				# print("#######################")


				if terminated_state is True :
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or False:
							self.total_episodes.append(self.rnn_episodes[i][k])
							self.rnn_episodes[i][k] = RNNEpisodeBuffer()
					self.env.reset(i)


			if local_step >= self.buffer_size:
				for i in range(self.num_slaves):
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or False:
							# print(str(i)+" "+str(k), end=' ')
							# print(len(self.episodes[i][k].getData()))
							self.total_episodes.append(self.rnn_episodes[i][k])
							self.rnn_episodes[i][k] = RNNEpisodeBuffer()
					self.env.reset(i)
				break
			for i in range(self.num_slaves):
				for j in range(self.num_agents):
					states[i*self.num_agents+j] = self.env.getState(i,j)
					hiddens[i*self.num_agents+j] = hiddens_forward[i*self.num_agents+j]

		self.env.endOfIteration()
		print('SIM : {}'.format(local_step))


	def optimizeSimulationNN(self):
		all_rnn_replay_buffer= np.array(self.rnn_buffer.buffer)
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

					# print(Tensor(batch.hidden_h).size())
					# print(Tensor(stack_hidden_h).size())
					# exit()
					# print(type(batch.s))
					# print(type(stack_s))
					# exit()

					num_layers =self.model[0].num_layers
					# stack_hidden = [[[]for x in range(4)], [[]for y in range(4)]]
					stack_hidden = [Tensor(np.reshape(stack_hidden_h, (num_layers,len(non_None_batch_list[timeStep]),-1))), 
										Tensor(np.reshape(stack_hidden_c, (num_layers,len(non_None_batch_list[timeStep]),-1)))]
					# stack_hidden = [None,None]

					# start = time.time()
					# if timeStep == 0:
					# 	stack_hidden = [Tensor(np.reshape(stack_hidden_h, (4,len(non_None_batch_segmented_transitions),-1))), 
					# 					Tensor(np.reshape(stack_hidden_c, (4,len(non_None_batch_segmented_transitions),-1)))]
					# 	# stack_hidden = tuple(stack_hidden)
					# else :
						# firstCat = True;
						# for k in non_None_batch_list[timeStep]:
						# 	if firstCat is True :
						# 		for l in range(len(non_None_batch_list[timeStep-1])) :
						# 			if non_None_batch_list[timeStep-1][l] == k:
						# 				for t in range(num_layers):
						# 					stack_hidden[0][t] = hidden_list[timeStep-1][0][t][l].unsqueeze(0)
						# 					stack_hidden[1][t] = hidden_list[timeStep-1][1][t][l].unsqueeze(0)
						# 				# print(hidden_list[timeStep-1][0][t][l].size())
						# 				# print(hidden_list[timeStep-1][0][t][l].unsqueeze(0).size())
						# 				# print(stack_hidden[0].size())
						# 				firstCat = False
						# 				break
						# 	else :
						# 		for l in range(len(non_None_batch_list[timeStep-1])) :
						# 			if non_None_batch_list[timeStep-1][l] == k:
						# 				for t in range(num_layers):
						# 					stack_hidden[0][t] = torch.cat((stack_hidden[0][t], hidden_list[timeStep-1][0][t][l].unsqueeze(0)),1)
						# 					stack_hidden[1][t] = torch.cat((stack_hidden[1][t], hidden_list[timeStep-1][1][t][l].unsqueeze(0)),1)

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

					# print(Tensor(stack_hidden[0]).size())
					a_dist,v,cur_stack_hidden = self.model[0].forward_rnn(Tensor(stack_s), stack_hidden)	
					# exit()

					hidden_list[timeStep] = list(cur_stack_hidden)
					# print(hidden_list[timeStep])

					# print(id(hidden_list[timeStep][0]))
					# print(id(cur_stack_hidden[0]))

					# hidden_list[timeStep] = copy.deep_copy(cur_stack_hidden)
					# print(id(hidden_list[timeStep]))
					# print(id(cur_stack_hidden))
					# print(hidden_list[timeStep][0].size())


					# if timeStep >= self.burn_in_size : 
					'''Critic Loss'''

					# if timeStep % 5 == 4:

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

						self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
						self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

						loss += loss_actor + loss_entropy + loss_critic

						if timeStep % 8 == 7:
							self.optimizer.zero_grad()

							# start = time.time()
							loss.backward(retain_graph=True)
							# print("time :", time.time() - start)

							for param in self.model[0].parameters():
								if param.grad is not None:
									param.grad.data.clamp_(-0.5, 0.5)
							self.optimizer.step()
			print('Optimizing sim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')
		# all_transitions = []
		# for rnn_replay_buffer in all_rnn_replay_buffer :
		# 	for single_transition in np.array(rnn_replay_buffer.buffer) :
		# 		all_transitions.append(single_transition)

		# for j in range(self.num_epochs):
		# 	np.random.shuffle(all_transitions)
		# 	for i in range(len(all_transitions)//self.batch_size):
		# 		transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
		# 		batch = Transition(*zip(*transitions))

		# 		stack_s =np.vstack(batch.s).astype(np.float32)
		# 		stack_a = np.vstack(batch.a).astype(np.float32)
		# 		stack_lp = np.vstack(batch.logprob).astype(np.float32)
		# 		stack_td = np.vstack(batch.TD).astype(np.float32)
		# 		stack_gae = np.vstack(batch.GAE).astype(np.float32)

		# 		a_dist,v = self.model[0](Tensor(stack_s))

		# 		# print(v.size())
		# 		# exit()

		# 		'''Critic Loss'''
		# 		loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()

		# 		'''Actor Loss'''
		# 		ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
		# 		stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+1E-5)
		# 		stack_gae = Tensor(stack_gae)
		# 		surrogate1 = ratio * stack_gae
		# 		surrogate2 = torch.clamp(ratio, min=1.0-self.clip_ratio, max=1.0+self.clip_ratio) * stack_gae
		# 		loss_actor = - torch.min(surrogate1, surrogate2).mean()

		# 		'''Entropy Loss'''
		# 		loss_entropy = - self.w_entropy * a_dist.entropy().mean()

		# 		self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
		# 		self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

		# 		loss = loss_actor + loss_entropy + loss_critic
	
				
			
		# 		# print(loss.cpu().detach().numpy().tolist())
		# 		# if np.any(np.isnan(loss.cpu().detach().numpy().tolist())):
		# 		# 	continue;

		# 		self.optimizer.zero_grad()
		# 		loss.backward(retain_graph=True)
		# 		# print(loss)
		# 		# exit()
		# 		for param in self.model[0].parameters():
		# 			if param.grad is not None:
		# 				param.grad.data.clamp_(-0.5, 0.5)
		# 		self.optimizer.step()
		# 	print('Optimizing sim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		# print('')


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
		print('||Noise                    : {:.3f}'.format(self.model[0].log_std.exp().mean()))		
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
		ppo.loadModel(args.model, 0)
	else:
		ppo.saveModel()
	print('num states: {}, num actions: {}'.format(ppo.env.getNumState(),ppo.env.getNumAction()))
	for i in range(ppo.max_iteration-5):
		ppo.train()
		rewards = ppo.evaluate()
		plot(rewards,'reward',0,False)





