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

sys.path.insert(0, "../")
import numpy as np
from pyvs import Env
from IPython import embed
import json
from Model import *

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

rolloutDir = "./rolloutData"

if not exists(rolloutDir):
	mkdir(rolloutDir)

NUM_TRANSITION = 100000


RNNEpisode = namedtuple('RNNEpisode', ('s','a','r','value','logprob'))

class RNNEpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def push(self, *args):
		self.data.append(RNNEpisode(*args))

	def getData(self):
		return self.data

RNNTransition = namedtuple('RNNTransition',('s','a','logprob','TD','GAE'))

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

class RolloutGenerator(object):
	def __init__(self):
		np.random.seed(seed = int(time.time()))
		self.num_slaves = 8
		self.num_agents = 6
		self.env = Env(self.num_agents)
		self.num_state = self.env.getNumState()
		self.num_action = self.env.getNumAction()

		self.policy = [None]*2
		self.policy[0] = SimulationNN(self.num_state, self.num_action)
		self.policy[1] = SimulationNN(self.num_state, self.num_action)
		

		self.num_epochs = 2
		self.num_evaluation = 0
		self.num_tuple_so_far = [0, 0]
		# self.num_episode = [0, 0]
		self.num_tuple = [0, 0]

		self.num_simulation_Hz = self.env.getSimulationHz()
		self.num_control_Hz = self.env.getControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.997
		self.lb = 0.95

		self.buffer_size = 48*1024
		self.batch_size = 512
		# self.trunc_size = 40
		# self.burn_in_size = 10
		# self.bptt_size = 10

		self.buffer = [ [None] for i in range(2)]
		self.buffer[0] = Buffer(100000)
		self.buffer[1] = Buffer(100000)

		self.model = [None]*self.num_slaves*self.num_agents
		for i in range(self.num_slaves*self.num_agents):
			# exit()
			self.model[i] = ActorCriticNN(self.num_state, self.num_action)
			if use_cuda:
				self.model[i].cuda()

		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio

		self.max_iteration = 50000

		self.w_entropy = 0.0001

		# self.loss_actor = 0.0
		self.loss_actor = [0.0, 0.0]

		self.loss_critic = [0.0, 0.0]

		self.sum_loss_actor = [0.0, 0.0]
		self.sum_loss_critic = [0.0, 0.0]

		self.rewards = []

		self.winRate = []

		self.sum_return = 0.0

		self.max_return = 0.0

		self.max_return_epoch = 1
		self.tic = time.time()

		self.episodes = [[RNNEpisodeBuffer() for y in range(self.num_agents)] for x in range(self.num_slaves)]
		self.indexToNetDic = {0:0, 1:1, 2:1,3:0,4:1,5:1}

		self.filecount = 0

		self.env.resets()

	def loadModel(self,path,index):
		self.model[index].load(path+'_'+str(self.indexToNetDic[index%self.num_agents])+'.pt')


	def saveModel(self):
		for i in range(2):
			self.model[i].save('../nn/current_'+str(i)+'.pt')

			if self.max_return_epoch == self.num_evaluation:
				self.model[i].save('../nn/max_'+str(i)+'.pt')
			if self.num_evaluation%20 == 0:
				self.model[i].save('../nn/'+str(self.num_evaluation)+'_'+str(i)+'.pt')



	def generateTransitions(self):
		self.total_episodes = [[] for i in range(2)]

		states = [None]*self.num_slaves*self.num_agents
		actions = [None]*self.num_slaves*self.num_agents
		rewards = [None]*self.num_slaves*self.num_agents
		logprobs = [None]*self.num_slaves*self.num_agents
		values = [None]*self.num_slaves*self.num_agents
		terminated = [False]*self.num_slaves*self.num_agents

		for i in range(self.num_slaves):
			for j in range(self.num_agents):
				states[i*self.num_agents+j] = self.env.getLocalState(i,j)

		learningTeam = random.randrange(0,2)
		'''Fixed to team 0'''
		learningTeam = 0
		# teamDic = {0: 0, 1: 1}
		teamDic = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

		local_step = 0
		counter = 0

		vsHardcodedFlags = [0]*self.num_slaves
		for i in range(self.num_slaves):
			randomNumber = random.randrange(0,100)
			# History
			if randomNumber < 5:
				vsHardcodedFlags[i] = 1
			# Hardcoding
			elif randomNumber < 10:
				vsHardcodedFlags[i] = 2
			# else : self-play

		print(vsHardcodedFlags)
		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')


			if self.num_evaluation > 20:
				curEvalNum = self.num_evaluation//20

				# clipedRandomNormal = 0
				# while clipedRandomNormal <= 0.5:
				# 	clipedRandomNormal = np.random.normal(curEvalNum, curEvalNum/2,1)[0]
				# 	if clipedRandomNormal > curEvalNum :
				# 		clipedRandomNormal = 2*curEvalNum - clipedRandomNormal;

				randomHistoryIndex = random.randrange(0,curEvalNum+1)

				for i in range(self.num_slaves):
					# prevPath = "../nn/"+clipedRandomNormal+".pt";

					for j in range(self.num_agents):
						if teamDic[j] != learningTeam:
							self.loadModel(str(20 * randomHistoryIndex), i*self.num_agents+j)
					# self.loadModel("current", i*self.num_agents+1)
			else :
				for i in range(self.num_slaves):
					for j in range(self.num_agents):
						if teamDic[j] != learningTeam:
							self.loadModel("0",i*self.num_agents+j)


			''' Scheduler Part '''
			for i in range(self.num_slaves):
				a_dist_slave = []
				v_slave = []
				for j in range(self.num_agents):
					if vsHardcodedFlags[i] == 0:
						a_dist_slave_agent,v_slave_agent = self.model[self.indexToNetDic[j]].forward(\
								Tensor([states[i*self.num_agents+j]]))
						a_dist_slave.append(a_dist_slave_agent)
						v_slave.append(v_slave_agent)
						actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze();					

					elif vsHardcodedFlags[i] == 1:
						if teamDic[j] == learningTeam:
							a_dist_slave_agent,v_slave_agent = self.model[self.indexToNetDic[j]].forward(\
								Tensor([states[i*self.num_agents+j]]))
						else :
							a_dist_slave_agent,v_slave_agent = self.model[i*self.num_agents+j].forward(\
								Tensor([states[i*self.num_agents+j]]))
						a_dist_slave.append(a_dist_slave_agent)
						v_slave.append(v_slave_agent)
						actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze();

					else :
						if teamDic[j] == learningTeam:
							a_dist_slave_agent,v_slave_agent = self.model[self.indexToNetDic[j]].forward(\
								Tensor([states[i*self.num_agents+j]]))
							a_dist_slave.append(a_dist_slave_agent)
							v_slave.append(v_slave_agent)
							actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy().squeeze().squeeze().squeeze();		
						else :
							actions[i*self.num_agents+j] = self.env.getHardcodedAction(i, j);

				for j in range(self.num_agents):
					if teamDic[j] == learningTeam or vsHardcodedFlags[i] == 0:
						logprobs[i*self.num_agents+j] = a_dist_slave[j].log_prob(Tensor(actions[i*self.num_agents+j]))\
							.cpu().detach().numpy().reshape(-1)[0];
						values[i*self.num_agents+j] = v_slave[j].cpu().detach().numpy().reshape(-1)[0];


				''' Set the Linear Actor state with scheduler action '''
				for j in range(self.num_agents):
					# if teamDic[j] == learningTeam:
					# print(actions[i*self.num_agents+j])

						# self.env.setLinearActorState(i, j, actions[i*self.num_agents+j])
					self.env.setAction(actions[i*self.num_agents+j], i, j);
					# else:

			self.env.stepsAtOnce()


			for i in range(self.num_slaves):
				nan_occur = False
				terminated_state = True
				for k in range(self.num_agents):
					if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
						rewards[i*self.num_agents+k] = self.env.getReward(i, k, True) 
						# print(rewards[i*self.num_agents+k], end=' ')
						if np.any(np.isnan(rewards[i*self.num_agents+k])):
							nan_occur = True
						if np.any(np.isnan(states[i*self.num_agents+k])) or np.any(np.isnan(actions[i*self.num_agents+k])):
							nan_occur = True

				if nan_occur is True:
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
							self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
							self.episodes[i][k] = RNNEpisodeBuffer()

					self.env.reset(i)


				if self.env.isTerminalState(i) is False:
					terminated_state = False
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
							self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k],\
								rewards[i*self.num_agents+k], values[i*self.num_agents+k], logprobs[i*self.num_agents+k])

							local_step += 1

				if terminated_state is True:
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
							self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k],\
								rewards[i*self.num_agents+k], values[i*self.num_agents+k], logprobs[i*self.num_agents+k])
							# print(str(i)+" "+str(k)+" "+str(rewards[i*self.num_agents+k]))
							self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
							self.episodes[i][k] = RNNEpisodeBuffer()

					self.env.reset(i)

			if local_step >= NUM_TRANSITION:
				for i in range(self.num_slaves):
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam or vsHardcodedFlags[i] == 0:
							self.total_episodes[self.indexToNetDic[k]].append(self.episodes[i][k])
							self.episodes[i][k] = RNNEpisodeBuffer()

					self.env.reset(i)
				break

			# get the updated state and updated hidden
			for i in range(self.num_slaves):
				for j in range(self.num_agents):
					states[i*self.num_agents+j] = self.env.getLocalState(i,j)

		self.env.endOfIteration()
		print('Rollout transitions : {}'.format(local_step))


import argparse
if __name__=="__main__":
	rollgen = RolloutGenerator()
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')
	parser.add_argument('-p','--policy',help='pretrained pollicy path')
	parser.add_argument('-iter','--iteration',help='num iterations')
	parser.add_argument('-n','--name',help='name of training setting')

	args =parser.parse_args()
	
	if args.model is not None:
		for k in range(rollgen.num_agents):
			rollgen.loadModel(args.model, k)
	else:
		print("ERROR : Please load the network")
		exit(0)

	rollgen.generateTransitions()