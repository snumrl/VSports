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
LOW_FREQUENCY = 3
HIGH_FREQUENCY = 30
device = torch.device("cuda" if use_cuda else "cpu")

nnCount = 2
baseDir = "../nn_lar_h"
nndir = baseDir + "/nn"+str(nnCount)

if not exists(baseDir):
    mkdir(baseDir)

if not exists(nndir):
	mkdir(nndir)


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

class RL(object):
	def __init__(self, motion_nn):
		np.random.seed(seed = int(time.time()))
		self.num_slaves = 16
		self.num_agents = 1
		self.env = Env(self.num_agents, motion_nn, self.num_slaves)
		self.num_state = self.env.getNumState()
		# self.num_action = self.env.getNumAction()
		self.num_policy = 1


		self.num_epochs = 4
		self.num_evaluation = 0
		self.num_tuple_so_far = [0, 0]
		# self.num_episode = [0, 0]
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

		#contact, finger, finger-ball
		self.num_action = [self.num_action_types, self.latent_size]


		self.num_h = len(self.num_action);

		self.buffer = [[ [None] for _ in range(self.num_policy)] for _ in range(self.num_h)];

		for h in range(self.num_h):
			for i in range(self.num_policy):
				self.buffer[h][i] = Buffer(100000)


		self.actionDecoders = [ VAEDecoder().to(device) for _ in range(self.num_action_types)]
		# for i in range(self.num_action_types):

		self.actionDecoders[0].load("vae_nn4/vae_action_decoder_"+str(0)+".pt")
		self.actionDecoders[1].load("vae_nn4/vae_action_decoder_"+str(3)+".pt")

		self.rms = RunningMeanStd(self.num_state-2)


		# self.buffer_0 = [ [None] for i in range(self.num_policy)]
		# for i in range(self.num_policy):
		# 	self.buffer_0[i] = Buffer(100000)


		# self.buffer_1 = [ [None] for i in range(self.num_policy)]
		# for i in range(self.num_policy):
		# 	self.buffer_1[i] = Buffer(100000)

		# self.buffer_2 = [ [None] for i in range(self.num_policy)]
		# for i in range(self.num_policy):
		# 	self.buffer_2[i] = Buffer(100000)



		# num action types 
		# self.num_action_0 = 6
		# self.num_action_1 = 4
		# self.num_action_2 = 8;

		# action type / root velocity, ball direction / (shooting) ball height, ball velocity, hand contact, fingerAngle

		# self.target_model_0 = [None]*self.num_policy
		# self.target_model_1 = [None]*self.num_policy
		# self.target_model_2 = [None]*self.num_policy

		self.target_model = [[[None] for _ in range(self.num_policy)] for _ in range(self.num_h)]


		acc_num_action = 0
		for h in range(self.num_h):
			for j in range(self.num_policy):
				if h== 0:
					self.target_model[h][j] = ActorCriticNN(self.num_state + acc_num_action, self.num_action[h], 
						log_std = 0.0, softmax = True, actionType = True)
				else:
					self.target_model[h][j] = ActorCriticNN(self.num_state + acc_num_action, self.num_action[h], 
						log_std = 0.0)

				acc_num_action += self.num_action[h]
				if use_cuda:
					self.target_model[h][j].cuda()


		# for i in range(self.num_policy):
		# 	self.target_model_1[i] = ActorCriticNN(self.num_state + self.num_action_0, self.num_action_1)
		# 	if use_cuda:
		# 		self.target_model_1[i].cuda()

		# for i in range(self.num_policy):
		# 	self.target_model_2[i] = ActorCriticNN(self.num_state + self.num_action_0 + self.num_action_1, self.num_action_2)
		# 	if use_cuda:
		# 		self.target_model_2[i].cuda()



		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio

		self.optimizer = [[[None] for _ in range(self.num_policy)] for _ in range(self.num_h)]

		# self.optimizer_0 = [None]*self.num_policy
		for h in range(self.num_h):
			for i in range(self.num_policy):
				self.optimizer[h][i] = optim.Adam(self.target_model[h][i].parameters(), lr=self.learning_rate)

		# embed()
		# exit(0)



		# self.optimizer_1 = [None]*self.num_policy
		# for i in range(self.num_policy):
		# 	self.optimizer_1[i] = optim.Adam(self.target_model_1[i].parameters(), lr=self.learning_rate)

		# self.optimizer_2 = [None]*self.num_policy
		# for i in range(self.num_policy):
		# 	self.optimizer_2[i] = optim.Adam(self.target_model_2[i].parameters(), lr=self.learning_rate)




		self.max_iteration = 50000

		self.w_entropy = 0.0001

		# self.loss_actor = [0.0 for _ in range(self.num_policy)]
		# self.loss_critic = [0.0 for _ in range(self.num_policy)]

		self.sum_loss_actor = [[0.0 for _ in range(self.num_policy)] for _ in range(self.num_h)] 
		self.sum_loss_critic = [[0.0 for _ in range(self.num_policy)] for _ in range(self.num_h)] 

		# self.sum_loss_actor_0 = [0.0 for _ in range(self.num_policy)]
		# self.sum_loss_critic_0 = [0.0 for _ in range(self.num_policy)]


		# self.sum_loss_actor_1 = [0.0 for _ in range(self.num_policy)]
		# self.sum_loss_critic_1 = [0.0 for _ in range(self.num_policy)]

		# self.sum_loss_actor_2 = [0.0 for _ in range(self.num_policy)]
		# self.sum_loss_critic_2 = [0.0 for _ in range(self.num_policy)]


		self.rewards = []

		self.sum_return = 0.0

		self.max_return = 0.0

		self.max_winRate = 0.0

		self.max_winRate_epoch = 0

		self.max_return_epoch = 1

		self.tic = time.time()

		# self.episodes_0 = [[RNNEpisodeBuffer() for y in range(self.num_agents)] for x in range(self.num_slaves)]
		# self.episodes_1 = [[RNNEpisodeBuffer() for y in range(self.num_agents)] for x in range(self.num_slaves)]
		# self.episodes_2 = [[RNNEpisodeBuffer() for y in range(self.num_agents)] for x in range(self.num_slaves)]

		self.episodes = [[[RNNEpisodeBuffer() for _ in range(self.num_agents)] for _ in range(self.num_slaves)] for _ in range(self.num_h)]
        
		self.indexToNetDic = {0:0, 1:0}

		self.filecount = 0

		self.env.slaveResets()


	def loadTargetModels(self,path,index):
		for h in range(self.num_h):
			self.target_model[h][self.indexToNetDic[index]].load(nndir+'/'+path+'_'+str(self.indexToNetDic[index%self.num_agents])+'_'+str(h)+'.pt')
		# self.target_model_0[self.indexToNetDic[index]].load(nndir+'/'+path+'_'+str(self.indexToNetDic[index%self.num_agents])+'_0.pt')
		# self.target_model_1[self.indexToNetDic[index]].load(nndir+'/'+path+'_'+str(self.indexToNetDic[index%self.num_agents])+'_1.pt')
		# self.target_model_2[self.indexToNetDic[index]].load(nndir+'/'+path+'_'+str(self.indexToNetDic[index%self.num_agents])+'_2.pt')
		self.rms.load(nndir+'/rms.ms')

	def saveModels(self):
		for i in range(self.num_policy):
			for h in range(self.num_h):
				self.target_model[h][i].save(nndir+'/'+'current_'+str(i)+'_'+str(h)+'.pt')
				# self.target_model_1[i].save(nndir+'/'+'current_'+str(i)+'_1.pt')
				# self.target_model_2[i].save(nndir+'/'+'current_'+str(i)+'_2.pt')

			if self.max_return_epoch == self.num_evaluation:
				for h in range(self.num_h):
					self.target_model[h][i].save(nndir+'/'+'max_'+str(i)+'_'+str(h)+'.pt')
				# self.target_model_0[i].save(nndir+'/'+'max_'+str(i)+'_0.pt')
				# self.target_model_1[i].save(nndir+'/'+'max_'+str(i)+'_1.pt')
				# self.target_model_2[i].save(nndir+'/'+'max_'+str(i)+'_2.pt')
			if self.num_evaluation%100 == 0:
				for h in range(self.num_h):
					self.target_model[h][i].save(nndir+'/'+str(self.num_evaluation)+'_'+str(i)+'_'+str(h)+'.pt')
				# self.target_model_0[i].save(nndir+'/'+str(self.num_evaluation)+'_'+str(i)+'_0.pt')
				# self.target_model_1[i].save(nndir+'/'+str(self.num_evaluation)+'_'+str(i)+'_1.pt')
				# self.target_model_2[i].save(nndir+'/'+str(self.num_evaluation)+'_'+str(i)+'_2.pt')
		self.rms.save(nndir+'/rms.ms')
		# f = open(nndir+'/rms.ms', 'w')
		# for i in range(len(self.rms.mean)):
		# 	f.write(str(self.rms.mean[i]))
		# 	f.write(" ")
		# 	f.write(str(self.rms.var[i]))
		# 	f.write("\n")
		# f.close()

	def arrayToOneHotVector(nparr):
		result = np.array(list(np.copy(nparr)))
		# resultVector = 

		# embed()
		# exit(0)

		for agent in range(len(nparr)):
			for slaves in range(len(nparr[agent])):
				maxIndex = 0
				maxValue = -100
				for i in range(len(nparr[agent][slaves])):
					result[agent][slaves][i] = 0.0
					if nparr[agent][slaves][i] > maxValue:
						maxValue = nparr[agent][slaves][i]
						maxIndex = i
				result[agent][slaves][maxIndex] = 1.0
		return result


	def generateTransitions(self):
		# self.total_episodes_0 = [[] for i in range(self.num_policy)]
		# self.total_episodes_1 = [[] for i in range(self.num_policy)]
		# self.total_episodes_2 = [[] for i in range(self.num_policy)]

		self.total_episodes = [ [[] for i in range(self.num_policy)] for _ in range(self.num_h)]

		states = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]
		actions = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]
		rewards = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]

		states_h = np.array([None for _ in range(self.num_h)])
		actions_h = np.array([[None for _ in range(self.num_agents)] for _ in range(self.num_h)])
		logprobs_h = np.array([[None for _ in range(self.num_agents)] for _ in range(self.num_h)])
		values_h = np.array([[None for _ in range(self.num_agents)] for _ in range(self.num_h)])

		# states_1 = [None]*self.num_slaves*self.num_agents
		# actions_1 = [None]*self.num_slaves*self.num_agents
		# logprobs_1 = [None]*self.num_slaves*self.num_agents
		# values_1 = [None]*self.num_slaves*self.num_agents


		# states_2 = [None]*self.num_slaves*self.num_agents
		# actions_2 = [None]*self.num_slaves*self.num_agents
		# logprobs_2 = [None]*self.num_slaves*self.num_agents
		# values_2 = [None]*self.num_slaves*self.num_agents



		terminated = [False]*self.num_slaves*self.num_agents

		# embed()
		# exit(0)
		for i in range(self.num_agents):
			for j in range(self.num_slaves):
				states[i][j] = self.env.getState(j,i).astype(np.float32)

		states = np.array(states)
		states[:,:,:-2] = self.rms.apply(states[:,:,:-2])
		mask = states[:,:,-2:]
		# embed()
		# exit(0)
		# states.transpose()
		# states = states.reshape(self.num_slaves, self.num_agents, -1)
		# states = np.transpose(states, (1,0,2))

		# states= states.astype(np.float32)
		# embed()
		# exit(0)

		learningTeam = 0
		# teamDic = {0: 0, 1: 1}
		teamDic = {0: 0, 1: 0}

		local_step = 0
		counter = 0

		def arrayToOneHotVectorWithConstraint(nparr):
			result = np.array(list(np.copy(nparr)))

			# resultVector = 

			# embed()
			# exit(0)

			for agent in range(len(nparr)):
				for slaves in range(len(nparr[agent])):
					maxIndex = 0
					maxValue = -100
					# embed()
					# exit()
					for i in range(len(nparr[agent][slaves])):
						result[agent][slaves][i] = 0.0
						if nparr[agent][slaves][i] > maxValue:
							maxValue = nparr[agent][slaves][i]
							maxIndex = i
					maxIndex = self.env.setActionType(maxIndex, slaves, agent)
					# print(maxIndex)
					result[agent][slaves][maxIndex] = 1.0
			return result


		# def arrayToOneHotVector(nparr):
		# 	result = np.array(list(np.copy(nparr)))
		# 	# resultVector = 

		# 	# embed()
		# 	# exit(0)

		# 	for agent in range(len(nparr)):
		# 		for slaves in range(len(nparr[agent])):
		# 			maxIndex = 0
		# 			maxValue = -100
		# 			for i in range(len(nparr[agent][slaves])):
		# 				result[agent][slaves][i] = 0.0
		# 				if nparr[agent][slaves][i] > maxValue:
		# 					maxValue = nparr[agent][slaves][i]
		# 					maxIndex = i
		# 			result[agent][slaves][maxIndex] = 1.0
		# 	return result



		def getActionTypeFromVector(vec):
			maxIndex = 0
			maxValue = -100
			for i in range(len(vec)):
				if vec[i] == 1:
					return i
			print("Something wrong in vec")
			print(vec)
			return 0


		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')


			# generate transition of first hierachy
			states_h[0] = states

			a_dist_slave = [None]*self.num_agents
			v_slave = [None]*self.num_agents
			for i in range(self.num_agents):
				if teamDic[i] == learningTeam:
					a_dist_slave_agent,v_slave_agent = self.target_model[0][self.indexToNetDic[i]].forward(\
						Tensor(states_h[0][i]))
					a_dist_slave[i] = a_dist_slave_agent
					v_slave[i] = v_slave_agent
					# print(actions_h)
					actions_h[0][i] = a_dist_slave[i].sample().cpu().detach().numpy().squeeze().squeeze();	
					# print(actions_h)

			# 	print(actions_h)
			# print(actions_h)

			for i in range(self.num_agents):
				if teamDic[i] == learningTeam:
					logprobs_h[0][i] = a_dist_slave[i].log_prob(Tensor(actions_h[0][i]))\
						.cpu().detach().numpy().reshape(-1);
					values_h[0][i] = v_slave[i].cpu().detach().numpy().reshape(-1);


			# embed()
			# exit(0)
			# generate transition of second hierachy
			# actions_0_oneHot = np.array(list(actions_h[0]))
			# start = time.time()
			actions_0_oneHot = arrayToOneHotVectorWithConstraint(actions_h[0])


			# print("time :", time.time() - start)

			# embed()
			# exit(0)
			for h in range(1,self.num_h):
				if h == 1:
					# embed()
					# exit(0)
					states_h[h] = np.concatenate((states_h[0], actions_0_oneHot), axis=2)
					# embed()
					# exit(0)
				else:
					# print("value h : {}".format(h))
					# embed()
					# exit(0)
					states_h[h] = np.concatenate((states_h[h-1], np.array(list(actions_h[h-1]))), axis=2)
				a_dist_slave = [None]*self.num_agents
				v_slave = [None]*self.num_agents
				for i in range(self.num_agents):
					if teamDic[i] == learningTeam:
						a_dist_slave_agent,v_slave_agent = self.target_model[h][self.indexToNetDic[i]].forward(\
							Tensor([states_h[h][i]]))
						a_dist_slave[i] = a_dist_slave_agent
						v_slave[i] = v_slave_agent
						actions_h[h][i] = a_dist_slave[i].sample().cpu().detach().numpy().squeeze().squeeze();		

				for i in range(self.num_agents):
					if teamDic[i] == learningTeam:
						logprobs_h[h][i] = a_dist_slave[i].log_prob(Tensor(actions_h[h][i]))\
							.cpu().detach().numpy().reshape(-1);
						values_h[h][i] = v_slave[i].cpu().detach().numpy().reshape(-1);
						# embed()
						# exit(0)




			# actions_h = np.array(list(actions_h))


			actions = actions_0_oneHot
			# embed()
			# exit(0)
			for h in range(1, self.num_h):
				# print(actions[0])
				actions = np.concatenate((actions, np.array(list(actions_h[h]))), axis = 2)
			# print(actions[0])


			actions = actions.astype(np.float32)
			# embed()
			# exit(0)
			# action : torch.Size([1, 16, 6])

			actionTypePart = actions[:,:,0:self.num_action_types]
			actionsDecodePart = actions[:,:,self.num_action_types:self.num_action_types+self.latent_size]
			# actionsRemainPart = actions[:,:,self.num_action_types+self.latent_size:]


			decodeShape = list(np.shape(actionsDecodePart))
			decodeShape[2] = 9
			actionsDecoded =np.empty(decodeShape,dtype=np.float32)

			for i in range(len(actionsDecodePart)):
				for j in range(len(actionsDecodePart[i])):
					# embed()
					# exit(0)
					curActionType = getActionTypeFromVector(actionTypePart[i][j])
					# embed()
					# exit(0)
					actionsDecoded[i][j] = self.actionDecoders[curActionType].decode(Tensor(actionsDecodePart[i][j])).cpu().detach().numpy()

			# print("time :", time.time() - start)

			envActions = actionsDecoded
			# envActions = np.concatenate((actionsDecoded, actionsRemainPart), axis=2)

			for i in range(self.num_agents):
				for j in range(self.num_slaves):
					# if np.any(np.isnan(actions[i*self.num_agents+j])):
						# print(states_1[i*self.num_agents+j])
						# print(actions_1[i*self.num_agents+j])
						# print(actions_2[i*self.num_agents+j])
						# print(actions[i*self.num_agents+j])
						# print("ShootArTraining nan action")
					# print("In python : ", end="")
					# print(actions[i][j])
					self.env.setAction(envActions[i][j], j, i);

			# exit(0)

			# actions = np.concatenate((actions_0, actions_1, actions_2), axis=1)

			# print("time :", time.time() - start)

			# start = time.time()
			self.env.stepsAtOnce()

			# print("time :", time.time() - start)

			nan_occur = [False]*self.num_slaves
			# for i in range(self.num_agents):
				# terminated_state = True
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
								for h in range(self.num_h):
									self.total_episodes[h][self.indexToNetDic[i]].append(self.episodes[h][j][i])
									self.episodes[h][j][i] = RNNEpisodeBuffer()

								# self.total_episodes_1[self.indexToNetDic[k]].append(self.episodes_1[i][k])
								# self.total_episodes_2[self.indexToNetDic[k]].append(self.episodes_2[i][k])
								# self.episodes_0[i][k] = RNNEpisodeBuffer()
								# self.episodes_1[i][k] = RNNEpisodeBuffer()
								# self.episodes_2[i][k] = RNNEpisodeBuffer()
						print("nan", file=sys.stderr)
						self.env.slaveReset(j)


					if self.env.isTerminalState(j) is False:
						for i in range(self.num_agents):
							if teamDic[i] == learningTeam:
								for h in range(self.num_h):
									self.episodes[h][j][i].push(states_h[h][i][j], actions_h[h][i][j],\
										rewards[i][j], values_h[h][i][j], logprobs_h[h][i][j])
								# self.episodes_1[i][k].push(states_1[i*self.num_agents+k], actions_1[i*self.num_agents+k],\
								# 	rewards[i*self.num_agents+k], values_1[i*self.num_agents+k], logprobs_1[i*self.num_agents+k])
								# self.episodes_2[i][k].push(states_2[i*self.num_agents+k], actions_2[i*self.num_agents+k],\
								# 	rewards[i*self.num_agents+k], values_2[i*self.num_agents+k], logprobs_2[i*self.num_agents+k])
								local_step += 1
					else:
						for i in range(self.num_agents):
							if teamDic[i] == learningTeam:
								for h in range(self.num_h):
									self.episodes[h][j][i].push(states_h[h][i][j], actions_h[h][i][j],\
										rewards[i][j], values_h[h][i][j], logprobs_h[h][i][j])


								for h in range(self.num_h):
									self.total_episodes[h][self.indexToNetDic[i]].append(self.episodes[h][j][i])
									self.episodes[h][j][i] = RNNEpisodeBuffer()
						self.env.slaveReset(j)


			if local_step >= self.buffer_size:
				for j in range(self.num_slaves):
					for i in range(self.num_agents):
						if teamDic[i] == learningTeam:
							for h in range(self.num_h):
								self.total_episodes[h][self.indexToNetDic[i]].append(self.episodes[h][j][i])
								self.episodes[h][j][i] = RNNEpisodeBuffer()

					self.env.slaveReset(j)
				break

			for i in range(self.num_agents):
				for j in range(self.num_slaves):
					states[i][j] = self.env.getState(j,i).astype(np.float32)
			states = np.array(states)
			states[:,:,:-2] = self.rms.apply(states[:,:,:-2])

		print('SIM : {}'.format(local_step))


	def computeTDandGAE(self):

		for h in range(self.num_h):
			for index in range(self.num_policy):
				self.buffer[h][index].clear()
				if index == 0:
					self.sum_return = 0.0
				for epi in self.total_episodes[h][index]:
					data = epi.getData()
					size = len(data)
					# print(size)
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


						rnn_replay_buffer = RNNReplayBuffer(10000)
						for i in range(size):

							rnn_replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

						self.buffer[h][index].push(rnn_replay_buffer)



		# for index in range(self.num_policy):
		# 	self.buffer_1[index].clear()
		# 	for epi in self.total_episodes_1[index]:
		# 		data = epi.getData()
		# 		size = len(data)
		# 		# print(size)
		# 		if size == 0:
		# 			continue
		# 		states, actions, rewards, values, logprobs = zip(*data)
		# 		values = np.concatenate((values, np.zeros(1)), axis=0)
		# 		advantages = np.zeros(size)
		# 		ad_t = 0

		# 		epi_return = 0.0
		# 		for i in reversed(range(len(data))):
		# 			epi_return += rewards[i]
		# 			delta = rewards[i] + values[i+1] * self.gamma - values[i]
		# 			ad_t = delta + self.gamma * self.lb * ad_t
		# 			advantages[i] = ad_t

		# 		if not np.isnan(epi_return):
		# 			TD = values[:size] + advantages
		# 			rnn_replay_buffer = RNNReplayBuffer(10000)
		# 			for i in range(size):

		# 				rnn_replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

		# 			self.buffer_1[index].push(rnn_replay_buffer)

		# for index in range(self.num_policy):
		# 	self.buffer_2[index].clear()
		# 	for epi in self.total_episodes_2[index]:
		# 		data = epi.getData()
		# 		size = len(data)
		# 		# print(size)
		# 		if size == 0:
		# 			continue
		# 		states, actions, rewards, values, logprobs = zip(*data)
		# 		values = np.concatenate((values, np.zeros(1)), axis=0)
		# 		advantages = np.zeros(size)
		# 		ad_t = 0

		# 		epi_return = 0.0
		# 		for i in reversed(range(len(data))):
		# 			epi_return += rewards[i]
		# 			delta = rewards[i] + values[i+1] * self.gamma - values[i]
		# 			ad_t = delta + self.gamma * self.lb * ad_t
		# 			advantages[i] = ad_t

		# 		if not np.isnan(epi_return):
		# 			TD = values[:size] + advantages
		# 			rnn_replay_buffer = RNNReplayBuffer(10000)
		# 			for i in range(size):

		# 				rnn_replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

		# 			self.buffer_2[index].push(rnn_replay_buffer)





			''' counting numbers '''
		for index in range(self.num_policy):
			self.num_episode = len(self.total_episodes[0][0])
			self.num_tuple[index] = 0
			for rnn_replay_buffer in self.buffer[0][index].buffer:
				self.num_tuple[index] += len(rnn_replay_buffer.buffer)
			self.num_tuple_so_far[index] += self.num_tuple[index]

	def optimizeNN_h(self, h = 0):
		for i in range(self.num_policy):
			self.sum_loss_actor[h][i] = 0.0
			self.sum_loss_critic[h][i] = 0.0

		for buff_index in range(self.num_policy):
			all_rnn_replay_buffer= np.array(self.buffer[h][buff_index].buffer)
			for j in range(self.num_epochs):
				all_segmented_transitions = []
				for rnn_replay_buffer in all_rnn_replay_buffer:
					rnn_replay_buffer_size = len(rnn_replay_buffer.buffer)
					for i in range(rnn_replay_buffer_size):
						all_segmented_transitions.append(rnn_replay_buffer.buffer[i])

				np.random.shuffle(all_segmented_transitions)
				for i in range(len(all_segmented_transitions)//self.batch_size):
					batch_segmented_transitions = all_segmented_transitions[i*self.batch_size:(i+1)*self.batch_size]

					# loss = Tensor(torch.zeros(1).cuda())

					batch = RNNTransition(*zip(*batch_segmented_transitions))

					stack_s = np.vstack(batch.s).astype(np.float32)
					stack_a = np.vstack(batch.a).astype(np.float32)
					stack_lp = np.vstack(batch.logprob).astype(np.float32)
					stack_td = np.vstack(batch.TD).astype(np.float32)
					stack_gae = np.vstack(batch.GAE).astype(np.float32)


					# num_layers = self.target_model[buff_index].num_layers

					a_dist,v = self.target_model[h][buff_index].forward(Tensor(stack_s))	
					
					# hidden_list[timeStep] = list(cur_stack_hidden)


					# embed()
					# exit(0)
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

					# self.loss_actor[buff_index] = loss_actor.cpu().detach().numpy().tolist()
					# self.loss_critic[buff_index] = loss_critic.cpu().detach().numpy().tolist()

					loss = loss_actor + loss_critic + loss_entropy
					self.optimizer[h][buff_index].zero_grad()



					# print(str(timeStep)+" "+str(offset))
					# start = time.time()
					loss.backward(retain_graph=True)

					# print("time :", time.time() - start)

					for param in self.target_model[h][buff_index].parameters():
						if param.grad is not None:
							param.grad.data.clamp_(-0.5, 0.5)

					self.optimizer[h][buff_index].step()
					self.sum_loss_actor[h][buff_index] += loss_actor.detach()*self.batch_size/self.num_epochs
					self.sum_loss_critic[h][buff_index] += loss_critic.detach()*self.batch_size/self.num_epochs
					loss_entropy = loss_entropy.detach()
					loss_actor = loss_actor.detach()
					loss_critic = loss_critic.detach()
					loss= loss.detach()


				print('Optimizing actor-critic nn_{} : {}/{}'.format(h, j+1,self.num_epochs),end='\r')
			print('')
		# print('')


	def train(self):
		frac = 1.0
		self.learning_rate = self.default_learning_rate*frac
		self.clip_ratio = self.default_clip_ratio*frac
		for i in range(self.num_policy):
			for h in range(self.num_h):
				for param_group in self.optimizer[h][i].param_groups:
					param_group['lr'] = self.learning_rate
					# if h == 0:
					# 	param_group['lr'] =  0.1*self.learning_rate

			# for param_group in self.optimizer_1[i].param_groups:
			# 	param_group['lr'] = self.learning_rate

			# for param_group in self.optimizer_2[i].param_groups:
			# 	param_group['lr'] = self.learning_rate

		self.generateTransitions()
		self.computeTDandGAE()
		# self.optimizeNN_h(0)
		self.optimizeModel()


	def optimizeModel(self):
		for h in range(self.num_h):
			self.optimizeNN_h(h)
		# self.optimizeNN_0()
		# self.optimizeNN_1()
		# self.optimizeNN_2()


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
			for h in range(self.num_h):
				print('||Avg Loss Actor {} {}         : {:.4f}'.format(h, i, self.sum_loss_actor[h][i]/self.num_tuple[i]))
				print('||Avg Loss Critic {} {}        : {:.4f}'.format(h, i, self.sum_loss_critic[h][i]/self.num_tuple[i]))
				print('||Noise {}                    : {:.3f}'.format(h, self.target_model[h][i].log_std.exp().mean()))		

			# print('||Avg Loss Actor 1 {}         : {:.4f}'.format(i, self.sum_loss_actor_1[i]/self.num_tuple[i]))
			# print('||Avg Loss Critic 1 {}        : {:.4f}'.format(i, self.sum_loss_critic_1[i]/self.num_tuple[i]))
			# print('||Noise 1                    : {:.3f}'.format(self.target_model[1][i].log_std.exp().mean()))		

			# print('||Avg Loss Actor 2 {}         : {:.4f}'.format(i, self.sum_loss_actor_2[i]/self.num_tuple[i]))
			# print('||Avg Loss Critic 2 {}        : {:.4f}'.format(i, self.sum_loss_critic_2[i]/self.num_tuple[i]))
			# print('||Noise 2                    : {:.3f}'.format(self.target_model[2][i].log_std.exp().mean()))		

			print('||Num Transition So far {}  : {}'.format(i, self.num_tuple_so_far[i]))
			print('||Num Transition {}         : {}'.format(i, self.num_tuple[i]))


		print('||Num Episode              : {}'.format(self.num_episode))
		print('||Avg Return per episode   : {:.3f}'.format(self.sum_return/self.num_episode))
		# print('||Avg Reward per transition: {:.3f}'.format(self.sum_return/self.num_tuple))
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

