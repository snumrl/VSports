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

import numpy as np
from pyvs import Env
from IPython import embed
import json
from Model import *
from ReinforcementLearning import *

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

LOW_FREQUENCY = 3
HIGH_FREQUENCY = 30

RNDEpisode = namedtuple('RNDEpisode', ('s','a','r','value','logprob','rnd_r'))

class RNDEpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def push(self, *args):
		self.data.append(RNDEpisode(*args))

	def getData(self):
		return self.data
def parallel_variance(avg_a, count_a, var_a, avg_b, count_b, var_b):
	    delta = avg_b - avg_a
	    m_a = var_a * (count_a - 1)
	    m_b = var_b * (count_b - 1)
	    M2 = m_a + m_b + delta ** 2 * count_a * count_b / (count_a + count_b)
	    return M2 / (count_a + count_b - 1)

class HExplorationRL(object):
	def __init__(self):
		np.random.seed(seed = int(time.time()))
		self.num_slaves = 8
		self.num_agents = 2
		self.env = Env(self.num_agents)
		self.num_state = self.env.getNumState()
		self.num_action = self.env.getNumAction()

		self.num_epochs = 10
		self.num_evaluation = 0
		self.num_tuple_so_far = [0, 0]
		self.num_tuple = [0, 0]

		self.num_simulation_Hz = self.env.getSimulationHz()
		self.num_control_Hz = self.env.getControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.997
		self.lb = 0.95

		self.buffer_size = 16*1024
		self.batch_size = 1024

		self.buffer = [ [None] for i in range(2)]
		self.buffer[0] = Buffer(100000)
		self.buffer[1] = Buffer(100000)

		self.model = [None]*self.num_slaves*self.num_agents
		for i in range(self.num_slaves*self.num_agents):
			self.model[i] = ActorCriticNN(self.num_state, self.num_action)
			if use_cuda:
				self.model[i].cuda()



		self.num_feature = 5
		self.target_rnd = RandomNN(self.num_state, self.num_feature)
		if use_cuda:
			self.target_rnd.cuda()
		self.target_rnd.save('../nn/target_rnd.pt')

		self.predictor_rnd = RandomNN(self.num_state, self.num_feature)
		if use_cuda:
			self.predictor_rnd.cuda()
		self.saveRNDModel()



		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio
		self.optimizer = [None]*2
		self.optimizer[0] = optim.Adam(self.model[0].parameters(), lr=self.learning_rate)
		self.optimizer[1] = optim.Adam(self.model[1].parameters(), lr=self.learning_rate)
		self.optimizer_rnd = optim.Adam(self.predictor_rnd.parameters(), lr=self.learning_rate)

		self.max_iteration = 50000

		self.w_entropy = 0.0001

		# self.loss_actor = 0.0
		self.loss_actor = [0.0, 0.0]

		self.loss_critic = [0.0, 0.0]

		self.sum_loss_actor = [0.0, 0.0]
		self.sum_loss_critic = [0.0, 0.0]


		self.loss_rnd = 0.0

		self.sum_loss_rnd = 0.0

		self.rewards = []

		self.winRate = []

		self.sum_return = 0.0

		self.max_return = 0.0

		self.max_return_epoch = 1
		self.tic = time.time()

		self.episodes = [[RNDEpisodeBuffer() for y in range(self.num_agents)] for x in range(self.num_slaves)]
		self.indexToNetDic = {0:0, 1:0}

		self.filecount = 0

		self.env.resets()

		self.total_ball_touch = 0
		self.ball_touch = []

		self.rnd_rewards_mean = 0.0
		self.rnd_rewards_std = 1.0

		self.total_rnd_count = 0


	def loadModel(self,path,index):
		self.model[index].load('../nn/'+path+'_'+str(self.indexToNetDic[index%self.num_agents])+'.pt')

	# def loadPolicy(self,path):
	# 	for i in range(2):
	# 		self.policy[i].load(path+"_"+str(i)+".pt")
	# 		self.model[i].policy.load_state_dict(self.policy[i].policy.state_dict())
	# 	self.saveModel()
	# 	self.winRate.append(self.evaluateModel())


	def saveModel(self):
		for i in range(1):
			self.model[i].save('../nn/current_'+str(i)+'.pt')
			if self.max_return_epoch == self.num_evaluation:
				self.model[i].save('../nn/max_'+str(i)+'.pt')
			if self.num_evaluation%20 == 0:
				self.model[i].save('../nn/'+str(self.num_evaluation)+'_'+str(i)+'.pt')

	def saveRNDModel(self):
		if self.num_evaluation%4 == 0:
			self.predictor_rnd.save('../nn/predictor_rnd_'+str(self.num_evaluation)+'.pt')
		self.predictor_rnd.save('../nn/predictor_rnd_current.pt')


	def getHardcodedAction(self, slave_index, agent_index):
		return np.array([0,0,0,-1])


	# def generateDefaultTransitions(self):
	# 	self.total_episodes = [[] for i in range(1)]

	# 	states = [None]*self.num_slaves*self.num_agents
	# 	actions = [None]*self.num_slaves*self.num_agents
	# 	rewards = [None]*self.num_slaves*self.num_agents
	# 	logprobs = [None]*self.num_slaves*self.num_agents
	# 	values = [None]*self.num_slaves*self.num_agents
	# 	terminated = [False]*self.num_slaves*self.num_agents

	# 	for i in range(self.num_slaves):
	# 		for j in range(self.num_agents):
	# 			states[i*self.num_agents+j] = self.env.getLocalState(i,j)

	# 	learningTeam = random.randrange(0,2)
	# 	'''Fixed to team 0'''
	# 	learningTeam = 0
	# 	teamDic = {0: 0, 1: 1}
	# 	# teamDic = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

	# 	local_step = 0
	# 	counter = 0

	# 	vsHardcodedFlags = [2]*self.num_slaves
	# 	# for i in range(self.num_slaves):
	# 	# 	randomNumber = random.randrange(0,100)
	# 	# 	# History
	# 	# 	if randomNumber < 5:
	# 	# 		vsHardcodedFlags[i] = 1
	# 	# 	# Hardcoding
	# 	# 	elif randomNumber < 10:
	# 	# 		vsHardcodedFlags[i] = 2
	# 	# 	# else : self-play

	# 	# print(vsHardcodedFlags)
	# 	while True:
	# 		counter += 1
	# 		if counter%10 == 0:
	# 			print('SIM : {}'.format(local_step),end='\r')


	# 		# if self.num_evaluation > 20:
	# 		# 	curEvalNum = self.num_evaluation//20

	# 		# 	randomHistoryIndex = random.randrange(0,curEvalNum+1)

	# 		# 	for i in range(self.num_slaves):
	# 		# 		# prevPath = "../nn/"+clipedRandomNormal+".pt";

	# 		# 		for j in range(self.num_agents):
	# 		# 			if teamDic[j] != learningTeam:
	# 		# 				self.loadModel(str(20 * randomHistoryIndex), i*self.num_agents+j)
	# 		# 		# self.loadModel("current", i*self.num_agents+1)
	# 		# else :
	# 		# 	for i in range(self.num_slaves):
	# 		# 		for j in range(self.num_agents):
	# 		# 			if teamDic[j] != learningTeam:
	# 		# 				self.loadModel("0",i*self.num_agents+j)


	# 		''' Scheduler Part '''
	# 		for i in range(self.num_slaves):
	# 			a_dist_slave = []
	# 			v_slave = []
	# 			for j in range(self.num_agents):
	# 				if vsHardcodedFlags[i] == 0:
	# 					a_dist_slave_agent,v_slave_agent = self.model[self.indexToNetDic[j]].forward(\
	# 							Tensor([states[i*self.num_agents+j]]))
	# 					a_dist_slave.append(a_dist_slave_agent)
	# 					v_slave.append(v_slave_agent)
	# 					actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze();					

	# 				elif vsHardcodedFlags[i] == 1:
	# 					if teamDic[j] == learningTeam:
	# 						a_dist_slave_agent,v_slave_agent = self.model[self.indexToNetDic[j]].forward(\
	# 							Tensor([states[i*self.num_agents+j]]))
	# 					else :
	# 						a_dist_slave_agent,v_slave_agent = self.model[i*self.num_agents+j].forward(\
	# 							Tensor([states[i*self.num_agents+j]]))
	# 					a_dist_slave.append(a_dist_slave_agent)
	# 					v_slave.append(v_slave_agent)
	# 					actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze();

	# 				else :
	# 					if teamDic[j] == learningTeam:
	# 						a_dist_slave_agent,v_slave_agent = self.model[self.indexToNetDic[j]].forward(\
	# 							Tensor([states[i*self.num_agents+j]]))
	# 						a_dist_slave.append(a_dist_slave_agent)
	# 						v_slave.append(v_slave_agent)
	# 						actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze();		
	# 					else :
	# 						actions[i*self.num_agents+j] = self.getHardcodedAction(i, j);

	# 			for j in range(self.num_agents):
	# 				if teamDic[j] == learningTeam or vsHardcodedFlags[i] == 0:
	# 					logprobs[i*self.num_agents+j] = a_dist_slave[j].log_prob(Tensor(actions[i*self.num_agents+j]))\
	# 						.cpu().detach().numpy().reshape(-1)[0];
	# 					values[i*self.num_agents+j] = v_slave[j].cpu().detach().numpy().reshape(-1)[0];


	# 			for j in range(self.num_agents):

	# 				self.env.setAction(actions[i*self.num_agents+j], i, j);

	# 		self.env.stepsAtOnce()


	# 		for i in range(self.num_slaves):
	# 			nan_occur = False
	# 			terminated_state = True
	# 			for k in range(self.num_agents):
	# 				if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
	# 					rewards[i*self.num_agents+k] = self.env.getReward(i, k, True)
	# 					if np.any(np.isnan(rewards[i*self.num_agents+k])):
	# 						nan_occur = True
	# 					if np.any(np.isnan(states[i*self.num_agents+k])) or np.any(np.isnan(actions[i*self.num_agents+k])):
	# 						nan_occur = True

	# 			if nan_occur is True:
	# 				for k in range(self.num_agents):
	# 					if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
	# 						self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
	# 						self.episodes[i][k] = RNDEpisodeBuffer()

	# 				self.env.reset(i)


	# 			if self.env.isTerminalState(i) is False:
	# 				terminated_state = False
	# 				for k in range(self.num_agents):
	# 					if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
	# 						self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k],\
	# 							rewards[i*self.num_agents+k], values[i*self.num_agents+k], logprobs[i*self.num_agents+k])

	# 						local_step += 1

	# 			if terminated_state is True:
	# 				for k in range(self.num_agents):
	# 					if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
	# 						self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k],\
	# 							rewards[i*self.num_agents+k], values[i*self.num_agents+k], logprobs[i*self.num_agents+k])
	# 						self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
	# 						self.episodes[i][k] = RNDEpisodeBuffer()

	# 				self.env.reset(i)

	# 		if local_step >= self.buffer_size:
	# 			for i in range(self.num_slaves):
	# 				for k in range(self.num_agents):
	# 					if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
	# 						self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
	# 						self.episodes[i][k] = RNDEpisodeBuffer()

	# 				self.env.reset(i)
	# 			break

	# 		# get the updated state and updated hidden
	# 		for i in range(self.num_slaves):
	# 			for j in range(self.num_agents):
	# 				states[i*self.num_agents+j] = self.env.getLocalState(i,j)


	# 	self.env.endOfIteration()
	# 	print('SIM : {}'.format(local_step))


	def generateTransitions(self, action_frequency):
		self.total_num_ball_touch = 0
		self.total_episodes = [[] for i in range(1)]

		states = [None]*self.num_slaves*self.num_agents
		actions = [None]*self.num_slaves*self.num_agents
		rewards = [None]*self.num_slaves*self.num_agents
		logprobs = [None]*self.num_slaves*self.num_agents
		values = [None]*self.num_slaves*self.num_agents
		terminated = [False]*self.num_slaves*self.num_agents
		actionCount = [0]*self.num_slaves
		rnd_rewards = [None]*self.num_slaves*self.num_agents

		actionNoise = [None]*self.num_slaves*self.num_agents

		def resetActionCount(arr, index):
			arr[index] = 30//action_frequency

		# for i in range(len(actionCount)):
		# 	resetActionCount(actionCount, i)


		for i in range(self.num_slaves):
			for j in range(self.num_agents):
				states[i*self.num_agents+j] = self.env.getLocalState(i,j)

		learningTeam = random.randrange(0,2)
		'''Fixed to team 0'''
		learningTeam = 0
		teamDic = {0: 0, 1: 1}
		# teamDic = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

		local_step = 0
		counter = 0

		vsHardcodedFlags = [2]*self.num_slaves


		# exploitationFlags = [0]*self.num_slaves
		# for i in range(self.num_slaves):
		# 	randomNumber = random.randrange(0, 5)
		# 	exploitationFlags[i] = randomNumber

		# print(exploitationFlags)
		# for i in range(self.num_slaves):
		# 	randomNumber = random.randrange(0,100)
		# 	# History
		# 	if randomNumber < 5:
		# 		vsHardcodedFlags[i] = 1
		# 	# Hardcoding
		# 	elif randomNumber < 10:
		# 		vsHardcodedFlags[i] = 2
		# 	# else : self-play

		# print(vsHardcodedFlags)
		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')


			# if self.num_evaluation > 20:
			# 	curEvalNum = self.num_evaluation//20

			# 	randomHistoryIndex = random.randrange(0,curEvalNum+1)

			# 	for i in range(self.num_slaves):
			# 		# prevPath = "../nn/"+clipedRandomNormal+".pt";

			# 		for j in range(self.num_agents):
			# 			if teamDic[j] != learningTeam:
			# 				self.loadModel(str(20 * randomHistoryIndex), i*self.num_agents+j)
			# 		# self.loadModel("current", i*self.num_agents+1)
			# else :
			# 	for i in range(self.num_slaves):
			# 		for j in range(self.num_agents):
			# 			if teamDic[j] != learningTeam:
			# 				self.loadModel("0",i*self.num_agents+j)


			''' Scheduler Part '''
			for i in range(self.num_slaves):
				a_dist_slave = [None]*self.num_agents
				v_slave = [None]*self.num_agents
				for j in range(self.num_agents):
					if vsHardcodedFlags[i] == 0:
						a_dist_slave_agent,v_slave_agent = self.model[self.indexToNetDic[j]].forward(\
								Tensor([states[i*self.num_agents+j]]))
						a_dist_slave[j] = a_dist_slave_agent
						v_slave[j] = v_slave_agent
						actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze();					

					elif vsHardcodedFlags[i] == 1:
						if teamDic[j] == learningTeam:
							a_dist_slave_agent,v_slave_agent = self.model[self.indexToNetDic[j]].forward(\
								Tensor([states[i*self.num_agents+j]]))
						else :
							a_dist_slave_agent,v_slave_agent = self.model[i*self.num_agents+j].forward(\
								Tensor([states[i*self.num_agents+j]]))
						a_dist_slave[j] = a_dist_slave_agent
						v_slave[j] = v_slave_agent
						actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze();

					# Here, only this case
					else :
						if teamDic[j] == learningTeam:
							a_dist_slave_agent,v_slave_agent = self.model[self.indexToNetDic[j]].forward(\
								Tensor([states[i*self.num_agents+j]]))
							a_dist_slave[j] = a_dist_slave_agent
							v_slave[j] = v_slave_agent
							# actions[i*self.num_agents+j] = a_dist_slave[j].loc.cpu().detach().numpy().squeeze().squeeze().squeeze();		
							# # if actionCount[i] == 0:
							# 	actionNoise[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze() - actions[i*self.num_agents+j]
								# print(actionNoise[i*self.num_agents+j])
							# if exploitationFlags[i] == 0:
							actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze()
						else :
							actions[i*self.num_agents+j] = self.getHardcodedAction(i, j);

				for j in range(self.num_agents):
					if teamDic[j] == learningTeam or vsHardcodedFlags[i] == 0:
						logprobs[i*self.num_agents+j] = a_dist_slave[j].log_prob(Tensor(actions[i*self.num_agents+j]))\
							.cpu().detach().numpy().reshape(-1)[0];
						values[i*self.num_agents+j] = v_slave[j].cpu().detach().numpy().reshape(-1)[0];


				for j in range(self.num_agents):
					self.env.setAction(actions[i*self.num_agents+j], i, j);

				if actionCount[i] == 0:
					resetActionCount(actionCount, i)
				actionCount[i] -= 1
				# print(actionCount)

			self.env.stepsAtOnce()


			for i in range(self.num_slaves):
				nan_occur = False
				terminated_state = True
				for k in range(self.num_agents):
					if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
						rewards[i*self.num_agents+k] = self.env.getReward(i, k, True)
						# target_feature = self.target_rnd.forward(Tensor([states[i*self.num_agents+k]])).cpu().detach().numpy().squeeze()
						# predictor_feature = self.predictor_rnd.forward(Tensor([states[i*self.num_agents+k]])).cpu().detach().numpy().squeeze()
						# diff_feature = Tensor(target_feature - predictor_feature).pow(2).sum()
						rnd_rewards[i*self.num_agents+k] = rewards[i*self.num_agents+k] 
						# rnd_rewards[i*self.num_agents+k] = rewards[i*self.num_agents+k]
						if np.any(np.isnan(rewards[i*self.num_agents+k])):
							nan_occur = True
						if np.any(np.isnan(states[i*self.num_agents+k])) or np.any(np.isnan(actions[i*self.num_agents+k])):
							nan_occur = True

				if nan_occur is True:
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
							self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
							self.episodes[i][k] = RNDEpisodeBuffer()
					self.total_num_ball_touch += self.env.getNumBallTouch(i)
					self.env.reset(i)
					actionCount[i] = 0


				if self.env.isTerminalState(i) is False:
					terminated_state = False
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
							self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k],\
								rewards[i*self.num_agents+k], values[i*self.num_agents+k], logprobs[i*self.num_agents+k], rnd_rewards[i*self.num_agents+k])

							local_step += 1

				if terminated_state is True:
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
							self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k],\
								rewards[i*self.num_agents+k], values[i*self.num_agents+k], logprobs[i*self.num_agents+k], rnd_rewards[i*self.num_agents+k])
							self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
							self.episodes[i][k] = RNDEpisodeBuffer()
					self.total_num_ball_touch += self.env.getNumBallTouch(i)
					self.env.reset(i)
					actionCount[i] = 0

			if local_step >= self.buffer_size:
				for i in range(self.num_slaves):
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
							self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
							self.episodes[i][k] = RNDEpisodeBuffer()
					self.total_num_ball_touch += self.env.getNumBallTouch(i)
					self.env.reset(i)
					actionCount[i] = 0
				break

			# get the updated state and updated hidden
			for i in range(self.num_slaves):
				for j in range(self.num_agents):
					states[i*self.num_agents+j] = self.env.getLocalState(i,j)


		# self.env.endOfIteration()
		print('SIM : {}'.format(local_step))

	

	def computeTDandGAE(self, is_final = False):
		# self.total_episodes = self.total_episodes + self.total_hindsight_episodes


		# compute mean and std of rnd rewards 
		iter_rnd_rewards = np.zeros(0)
		for index in range(1):
			for epi in self.total_episodes[index]:
				data = epi.getData()
				size = len(data)
				# print(size)
				if size == 0:
					continue
				states, actions, rewards, values, logprobs, rnd_rewards = zip(*data)
				iter_rnd_rewards = np.concatenate((rnd_rewards, iter_rnd_rewards), axis = 0)

		# iter_rnd_num = len(iter_rnd_rewards)
		# iter_rnd_mean = np.mean(iter_rnd_rewards)
		# iter_rnd_std = np.std(iter_rnd_rewards)

		# if self.total_rnd_count == 0:
		# 	self.total_rnd_count = iter_rnd_num
		# 	self.rnd_rewards_mean = iter_rnd_mean
		# 	self.rnd_rewards_std = iter_rnd_std
		# else:
		# 	self.rnd_rewards_std = parallel_variance(self.rnd_rewards_mean, self.total_rnd_count, self.rnd_rewards_std,
		# 											 iter_rnd_mean, iter_rnd_num, iter_rnd_std)
		# 	self.total_rnd_count += iter_rnd_num
		# 	self.rnd_rewards_mean = self.rnd_rewards_mean + (iter_rnd_mean - self.rnd_rewards_mean) * iter_rnd_num / self.total_rnd_count
		# print(self.rnd_rewards_mean)
		# print(self.rnd_rewards_std)


		'''Scheduler'''
		for index in range(1):
			self.buffer[index].clear()
			self.sum_return = 0.0
		# for i in range
			for epi in self.total_episodes[index]:
				data = epi.getData()
				size = len(data)
				# print(size)
				if size == 0:
					continue
				states, actions, rewards, values, logprobs, rnd_rewards = zip(*data)
				values = np.concatenate((values, np.zeros(1)), axis=0)
				advantages = np.zeros(size)
				ad_t = 0

				# rnd_rewards = (rnd_rewards)/(self.rnd_rewards_std+1E-5)
				# print(rnd_rewards)
				# rewards += rnd_rewards

				epi_return = 0.0
				for i in reversed(range(len(data))):
					epi_return += rewards[i]
					delta = rewards[i] + values[i+1] * self.gamma - values[i]
					ad_t = delta + self.gamma * self.lb * ad_t
					advantages[i] = ad_t

				if not np.isnan(epi_return):
					self.sum_return += epi_return
					TD = values[:size] + advantages


					rnn_replay_buffer = RNNReplayBuffer(10000)
					for i in range(size):

						# if TD[i] >= 0.1 or TD[i] <= -0.1:
						# 	rnn_replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

						# elif random.randrange(0,3) == 0:
						rnn_replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i])
						# x = rnn_replay_buffer[i]
						# j = json.dumps(x._asdict())
						# print(j)
						# exit(0)
					self.buffer[index].push(rnn_replay_buffer)
					# x = rnn_replay_buffer
					# i = 0
					# # print(x.buffer[i].TD)
					# if size <= 120:
					# 	while i < size:
					# 		# if abs(x.buffer[i].TD) > 9.0:
					# 		startTime = max(0, i)
					# 		endTime = min(size-1, i+120)

					# 		# j = json.dumps(x._asdict())
					# 		f = open("../data/high_TD_episode_"+str(self.filecount)+".txt", 'w')
					# 		f.write(str(endTime-startTime+1))
					# 		f.write("\n")

					# 		for line in range(startTime, endTime+1):
					# 			for line_s in range(len(x.buffer[line].s)):
					# 				f.write(str(x.buffer[line].s[line_s]))
					# 				f.write(" ")
					# 			# f.write(str(x.buffer[line].s))
					# 			# f.write(" ")
					# 			f.write(str(x.buffer[line].TD))
					# 			f.write("\n")
					# 		# print(j)

					# 		self.filecount += 1
					# 		if self.filecount == 40:
					# 			exit(0)
					# 		f.close()
					# 		self.buffer[index].push(rnn_replay_buffer)
					# 		i = endTime
					# 		i+=1


			''' counting numbers '''
			self.num_episode = len(self.total_episodes[index])
			self.num_tuple[index] = 0
			for rnn_replay_buffer in self.buffer[index].buffer:
				self.num_tuple[index] += len(rnn_replay_buffer.buffer)
			if is_final == True:
				self.num_tuple_so_far[index] += self.num_tuple[index]

	def exploreHighTDTransitions(self):
		sample_ratio = 1.0/900.0
		replay_time = 120


		for buff_index in range(1):
			replay_buffers = np.array(self.buffer[buff_index].buffer)
			index_array = []
			TD_array = []
			for replay_buffer_index in range(len(replay_buffers)):
				for transition_index in range(len(replay_buffers[replay_buffer_index].buffer)):
					transition = replay_buffers[replay_buffer_index].buffer[transition_index]
					index_array.append((replay_buffer_index, transition_index))	
					TD_array.append(pow(transition.TD, 2))
			num_samples = int(self.num_tuple[0] * sample_ratio)
			# print(num_samples)
			sampled_indices = random.choices(population=index_array, weights=TD_array, k = num_samples)
			sample_iter = 0
			for sampled_index in sampled_indices:
				sampled_epi = replay_buffers[sampled_index[0]]
				sampled_transition = sampled_epi.buffer[sampled_index[1]]
				print('{:.2f}'.format(sampled_transition.TD), end= " ")
			print("")

			for sampled_index in sampled_indices:
				sampled_epi = replay_buffers[sampled_index[0]]
				sampled_startpoint = max(sampled_index[1] - 60, 0)
				sampled_start_transition = sampled_epi.buffer[sampled_startpoint]
				# sampled_transition = sampled_epi.buffer[sampled_index[1]]
				# print(sampled_transition.TD)
				# replay_number = 5
				for i in range(self.num_slaves):
					self.env.reconEnvFromState(i, buff_index, sampled_start_transition.s)
				# print(sampled_start_transition.s)

				states = [None]*self.num_slaves*self.num_agents
				actions = [None]*self.num_slaves*self.num_agents
				rewards = [None]*self.num_slaves*self.num_agents
				logprobs = [None]*self.num_slaves*self.num_agents
				values = [None]*self.num_slaves*self.num_agents
				terminated = [False]*self.num_slaves*self.num_agents
				actionNoise = [None]*self.num_slaves*self.num_agents

				for i in range(self.num_slaves):
					for j in range(self.num_agents):
						states[i*self.num_agents+j] = self.env.getLocalState(i,j)

				# print(states[0*self.num_agents+0])
				# exit(0)
				# self.exploration_episodes = [[] for i in range(2)]

				learningTeam = 0
				teamDic = {0: 0, 1: 1}
				counter = 0
				local_step = 0

				# exploitationFlags = [0]*self.num_slaves
				# for i in range(self.num_slaves):
				# 	randomNumber = random.randrange(0, 5)
				# 	exploitationFlags[i] = randomNumber



				while True:
					if counter%10 == 0:
						print('Exploration SIM : {}'.format(local_step + replay_time*self.num_slaves*sample_iter),end='\r')
					for i in range(self.num_slaves):
						a_dist_slave = [None]*self.num_agents
						v_slave = [None]*self.num_agents
						for j in range(self.num_agents):
							if teamDic[j] == learningTeam:
								a_dist_slave_agent,v_slave_agent = self.model[self.indexToNetDic[j]].forward(\
									Tensor([states[i*self.num_agents+j]]))
								a_dist_slave[j] = a_dist_slave_agent
								v_slave[j] = v_slave_agent
								# actions[i*self.num_agents+j] = a_dist_slave[j].loc.cpu().detach().numpy().squeeze().squeeze().squeeze();		
								# actionNoise[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze() - actions[i*self.num_agents+j]
									# print(actionNoise[i*self.num_agents+j])
								# if exploitationFlags[i] != 0:
								# actions[i*self.num_agents+j] += actionNoise[i*self.num_agents+j]
								actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze()
							else :
								actions[i*self.num_agents+j] = self.getHardcodedAction(i, j);

						for j in range(self.num_agents):
							if teamDic[j] == learningTeam:
								logprobs[i*self.num_agents+j] = a_dist_slave[j].log_prob(Tensor(actions[i*self.num_agents+j]))\
									.cpu().detach().numpy().reshape(-1)[0];
								values[i*self.num_agents+j] = v_slave[j].cpu().detach().numpy().reshape(-1)[0];

						for j in range(self.num_agents):
							self.env.setAction(actions[i*self.num_agents+j], i, j);

					self.env.stepsAtOnce()

					for i in range(self.num_slaves):
						nan_occur = False
						terminated_state = True

						for k in range(self.num_agents):
							rewards[i*self.num_agents+k] = self.env.getReward(i, k, True)
							target_feature = self.target_rnd.forward(Tensor([states[i*self.num_agents+k]])).cpu().detach().numpy().squeeze()
							predictor_feature = self.predictor_rnd.forward(Tensor([states[i*self.num_agents+k]])).cpu().detach().numpy().squeeze()
							diff_feature = Tensor(target_feature - predictor_feature).pow(2).sum()
							rnd_rewards[i*self.num_agents+k] = diff_feature.cpu().detach().numpy()
							# rnd_rewards[i*self.num_agents+k] = rewards[i*self.num_agents+k]
							if np.any(np.isnan(rewards[i*self.num_agents+k])):
								nan_occur = True
							if np.any(np.isnan(states[i*self.num_agents+k])) or np.any(np.isnan(actions[i*self.num_agents+k])):
								nan_occur = True

						if nan_occur is True:
							for k in range(self.num_agents):
								if teamDic[k] == learningTeam:
									self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
									self.episodes[i][k] = RNDEpisodeBuffer()
							self.total_num_ball_touch += self.env.getNumBallTouch(i)
							self.env.reset(i)

						if self.env.isTerminalState(i) is False:
							terminated_state = False
							for k in range(self.num_agents):
								if teamDic[k] == learningTeam:
									self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k],\
										rewards[i*self.num_agents+k], values[i*self.num_agents+k], logprobs[i*self.num_agents+k])
									local_step += 1

						if terminated_state is True:
							for k in range(self.num_agents):
								if teamDic[k] == learningTeam:
									self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k],\
										rewards[i*self.num_agents+k], values[i*self.num_agents+k], logprobs[i*self.num_agents+k])
									self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
									self.episodes[i][k] = RNDEpisodeBuffer()
									local_step += 1
							self.total_num_ball_touch += self.env.getNumBallTouch(i)
							self.env.reset(i)
					# for i in range(self.num_slaves):
					# 	for k in range(self.num_agents):
					# 		if teamDic[k] == learningTeam:
					# 			print(len(self.episodes[i][k].getData()))
					counter += 1
					if counter >= replay_time:
						for i in range(self.num_slaves):
							for k in range(self.num_agents):
								if teamDic[k] == learningTeam:
									# print(len(self.episodes[i][k].getData()))
									if len(self.episodes[i][k].getData()) == 120:
										self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
									self.episodes[i][k] = RNDEpisodeBuffer()
							self.total_num_ball_touch += self.env.getNumBallTouch(i)
							self.env.reset(i)
						break

					for i in range(self.num_slaves):
						for j in range(self.num_agents):
							states[i*self.num_agents+j] = self.env.getLocalState(i,j)

				sample_iter += 1

			print('')



	def optimizeSchedulerNN(self):
		for i in range(1):
			self.sum_loss_actor[i] = 0.0
			self.sum_loss_critic[i] = 0.0
		self.sum_loss_rnd = 0.0

		for buff_index in range(1):
			all_rnn_replay_buffer= np.array(self.buffer[buff_index].buffer)
			for j in range(self.num_epochs):
				all_segmented_transitions = []
				for rnn_replay_buffer in all_rnn_replay_buffer:
					rnn_replay_buffer_size = len(rnn_replay_buffer.buffer)

					for i in range(rnn_replay_buffer_size):
						all_segmented_transitions.append(rnn_replay_buffer.buffer[i])
				# print(len(all_segmented_transitions))
				np.random.shuffle(all_segmented_transitions)
				for i in range(len(all_segmented_transitions)//self.batch_size):
					batch_segmented_transitions = all_segmented_transitions[i*self.batch_size:(i+1)*self.batch_size]

					loss = Tensor(torch.zeros(1).cuda())

					batch = RNNTransition(*zip(*batch_segmented_transitions))

					stack_s = np.vstack(batch.s).astype(np.float32)
					stack_a = np.vstack(batch.a).astype(np.float32)
					stack_lp = np.vstack(batch.logprob).astype(np.float32)
					stack_td = np.vstack(batch.TD).astype(np.float32)
					stack_gae = np.vstack(batch.GAE).astype(np.float32)

					num_layers = self.model[buff_index].num_layers

					a_dist,v = self.model[buff_index].forward(Tensor(stack_s))	
					
					loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()
					# loss_critic = ((v-Tensor(stack_td)).pow(2)-(v-Tensor(stack_td)).pow(2).mean())/((v-Tensor(stack_td)).pow(2).std()+1E-5)
					# print(loss_critic)
					'''Actor Loss'''
					ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
					stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+1E-5)
					stack_gae = Tensor(stack_gae)
					surrogate1 = ratio * stack_gae
					surrogate2 = torch.clamp(ratio, min=1.0-self.clip_ratio, max=1.0+self.clip_ratio) * stack_gae

					loss_actor = - torch.min(surrogate1, surrogate2).mean()

					'''Entropy Loss'''
					loss_entropy = - self.w_entropy * a_dist.entropy().mean()
					# print(loss_actor, end = ", ")
					# print(loss_critic, end = ", ")
					# print(loss_entropy)

					self.loss_actor[buff_index] = loss_actor.cpu().detach().numpy().tolist()
					self.loss_critic[buff_index] = loss_critic.cpu().detach().numpy().tolist()

					loss = loss_actor + loss_critic + loss_entropy
					self.optimizer[buff_index].zero_grad()


					# start = time.time()
					loss.backward(retain_graph=True)

					# print("time :", time.time() - start)

					# for param in self.model[buff_index].parameters():
						# if param.grad is not None:
							# print(param.grad.data)
							# param.grad.data.clamp_(-0.5, 0.5)

					self.optimizer[buff_index].step()
					self.sum_loss_actor[buff_index] += self.loss_actor[buff_index]*self.batch_size/self.num_epochs
					self.sum_loss_critic[buff_index] += self.loss_critic[buff_index]*self.batch_size/self.num_epochs



					# # RND optimizer

					# target_feature = self.target_rnd.forward(Tensor(stack_s)).detach()
					# predictor_feature = self.predictor_rnd.forward(Tensor(stack_s))

					# loss_rnd = self.num_feature * (target_feature - predictor_feature).pow(2).mean()
					# self.optimizer_rnd.zero_grad()

					# loss_rnd.backward(retain_graph=True)

					# # for param in self.predictor_rnd.parameters():
					# # 	if param.grad is not None:
					# # 		param.grad.data.clamp_(-0.5, 0.5)

					# self.optimizer_rnd.step()

					# self.loss_rnd = loss_rnd.cpu().detach().numpy().tolist()
					# self.sum_loss_rnd += self.loss_rnd*self.batch_size/self.num_epochs

				print('Optimizing actor-critic nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')



	def optimizeModel(self):
		self.computeTDandGAE(True)
		self.optimizeSchedulerNN()

	def train(self):
		frac = 1.0
		self.learning_rate = self.default_learning_rate*frac
		self.clip_ratio = self.default_clip_ratio*frac
		for i in range(2):
			for param_group in self.optimizer[i].param_groups:
				param_group['lr'] = self.learning_rate

		for param_group in self.optimizer_rnd.param_groups:
			param_group['lr'] = self.learning_rate
		# self.generateDefaultTransitions()
		self.generateTransitions(HIGH_FREQUENCY)
		# self.computeTDandGAE(False)
		# self.exploreHighTDTransitions()
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
		for i in range(1):
			if self.num_tuple[i]is 0:
				self.num_tuple[i] = 1
		if self.max_return < self.sum_return/self.num_tuple[0]:
			self.max_return = self.sum_return/self.num_tuple[0]
			self.max_return_epoch = self.num_evaluation

		# if self.num_evaluation%20 == 0:
		# 	self.winRate.append(self.evaluateModel())

		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,h,m,s))
		print('||--------------ActorCriticNN------------------')
		print('||Avg Loss Actor 0             : {:.4f}'.format(self.sum_loss_actor[0]/self.num_tuple[0]))
		print('||Avg Loss Critic 0            : {:.4f}'.format(self.sum_loss_critic[0]/self.num_tuple[0]))
		# print('||Loss Actor               : {:.4f}'.format(self.loss_actor))
		# print('||Loss Critic              : {:.4f}'.format(self.loss_critic))
		# print('||Noise                    : {:.3f}'.format(self.model[0].log_std.exp().mean()))		
		print('||Num Transition So far 0      : {}'.format(self.num_tuple_so_far[0]))
		print('||Num Transition 0             : {}'.format(self.num_tuple[0]))
		print('||Num Episode                  : {}'.format(self.num_episode))
		print('||Avg Return per tuple         : {:.4f}'.format(self.sum_return/self.num_tuple[0]))
		# print('||Avg Reward per transition: {:.3f}'.format(self.sum_return/self.num_tuple))
		print('||Avg Step per episode 0       : {:.1f}'.format(self.num_tuple[0]/self.num_episode))
		print('||Max Avg Retun So far         : {:.3f} at #{}'.format(self.max_return,self.max_return_epoch))
		print('||Num Ball touch per 1 second  : {:.3f}'.format(30.0 * self.total_num_ball_touch/self.num_tuple[0]))
		# print('||Current Win Rate         : {:.3f}'.format(self.winRate[-1]))

		print('||Mean RND Rewards             : {:.4f}'.format(self.rnd_rewards_mean))
		print('||Std RND Rewards              : {:.4f}'.format(self.rnd_rewards_std))



		self.rewards.append(self.sum_return/self.num_tuple[0])
		self.ball_touch.append(30.0 * self.total_num_ball_touch/self.num_tuple[0])
		
		self.saveModel()
		self.saveRNDModel()
		print('=============================================')
		return np.array(self.rewards), np.array(self.ball_touch)


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

	# plt.show()
	if ylim:
		plt.ylim([0,1])

	fig.canvas.draw()
	fig.canvas.flush_events()

def plot_ball(y,title,num_fig=1,ylim=True):
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
	plt.axhline(y=0.0, color='k', linewidth=1)
	plt.axhline(y=1.0, color='k', linewidth=1)

	# plt.show()
	if ylim:
		plt.ylim([0,1])

	fig.canvas.draw()
	fig.canvas.flush_events()



import argparse
if __name__=="__main__":
	rl = HExplorationRL()
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')
	parser.add_argument('-p','--policy',help='pretrained pollicy path')
	parser.add_argument('-iter','--iteration',help='num iterations')
	parser.add_argument('-n','--name',help='name of training setting')

	args =parser.parse_args()
	
	graph_name = ''
	
	if args.model is not None:
		for k in range(rl.num_agents):
			rl.loadModel(args.model, k)

	if args.name is not None:
		graph_name = args.name

	# if args.policy is not None:
	# 	rl.loadPolicy(args.policy)
	if args.iteration is not None:
		rl.num_evaluation = int(args.iteration)
		# for i in range(int(args.iteration)):
		# 	rl.env.endOfIteration()
	else:
		rl.saveModel()
	print('num states: {}, num actions: {}'.format(rl.env.getNumState(),rl.env.getNumAction()))
	for i in range(5000000):
		rl.train()
		rewards, ball_touch = rl.evaluate()
		plot(rewards, graph_name + 'Reward',0,False)
		plot_ball(ball_touch, 'Ball Touch Per Second',1,False)
		# plot_winrate(winRate, graph_name + 'vs Hardcoded Winrate',1,False)

