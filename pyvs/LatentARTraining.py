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
from VAE import VAEDecoder, VAEEncoder

import numpy as np

from pyvs import Env

from IPython import embed
import json
from Model import *
from pathlib import Path
from tensorboardX import SummaryWriter
summary = SummaryWriter()


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
LOW_FREQUENCY = 3
HIGH_FREQUENCY = 30
device = torch.device("cuda" if use_cuda else "cpu")

nnCount = 37
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

	def pop(self):
		self.data.pop()

	def popleft(self):
		self.data.pop(0)

	def getData(self):
		return self.data

	def getLastData(self):
		return self.data[len(self.data)-1]

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


		self.num_simulation_Hz = self.env.getSimulationHz()
		self.num_control_Hz = self.env.getControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.999
		self.lb = 0.95

		self.buffer_size = 16*1024
		self.batch_size = 2*512

		self.num_action_types = 5
		self.latent_size = 4

		self.resetDuration = self.env.getResetDuration()
		self.typeFreq = self.env.getTypeFreq()

		#contact, finger, finger-ball
		self.num_action = [self.num_action_types, self.latent_size]


		self.num_h = len(self.num_action);

		self.num_tuple_so_far = [[0, 0] for _ in range(self.num_h)]
		self.num_tuple = [[0, 0] for _ in range(self.num_h)]
		self.num_tutorial_tuple = [[0, 0] for _ in range(self.num_h)]

		self.buffer = [[ [None] for _ in range(self.num_policy)] for _ in range(self.num_h)];

		for h in range(self.num_h):
			for i in range(self.num_policy):
				self.buffer[h][i] = Buffer(100000)

		self.tutorial_buffer = [[ [None] for _ in range(self.num_policy)] for _ in range(self.num_h)];

		for h in range(self.num_h):
			for i in range(self.num_policy):
				self.tutorial_buffer[h][i] = Buffer(100000)



		self.actionDecoders = [ VAEDecoder().to(device) for _ in range(self.num_action_types)]
		self.actionEncoder = VAEEncoder().to(device)

		for i in range(self.num_action_types):
			self.actionDecoders[i].load("vae_nn_sep_0/vae_action_decoder_"+str(i)+".pt")

		self.actionEncoder.load("vae_nn_sep_0/vae_action_encoder.pt")

		self.rms = RunningMeanStd(self.num_state-5)

		self.num_c = 4

		self.stdScale = 1.0

		self.target_model = [[[None] for _ in range(self.num_policy)] for _ in range(self.num_h)]


		acc_num_action = 0
		self.num_hidden = 0
		for h in range(self.num_h):
			for j in range(self.num_policy):
				if h== 0:
					self.target_model[h][j] = ActorCriticNN(self.num_state, self.num_action[h], 
						log_std = 0.0, softmax = True, actionMask = True)
				else:
					self.target_model[h][j] = ActorCriticNN(self.num_state + self.num_action[0], self.num_action[h], 
						log_std = 0.0)

				acc_num_action += self.num_action[h]
				if use_cuda:
					self.target_model[h][j].cuda()

		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio

		self.optimizer = [[[None] for _ in range(self.num_policy)] for _ in range(self.num_h)]

		for h in range(self.num_h):
			for i in range(self.num_policy):
				self.optimizer[h][i] = optim.Adam(self.target_model[h][i].parameters(), lr=self.learning_rate)


		self.max_iteration = 50000

		self.w_entropy = 0.0001

		self.sum_loss_actor = [[0.0 for _ in range(self.num_policy)] for _ in range(self.num_h)] 
		self.sum_loss_critic = [[0.0 for _ in range(self.num_policy)] for _ in range(self.num_h)] 



		self.sum_tutorial_loss_critic = [[0.0 for _ in range(self.num_policy)] for _ in range(self.num_h)] 


		self.rewards = []
		self.numSteps = []
		self.num_correct_throwings = []

		self.sum_return = 0.0

		self.max_return = -10.0

		self.max_winRate = 0.0

		self.max_winRate_epoch = 0

		self.max_return_epoch = 1

		self.tic = time.time()

		self.episodes = [[[RNNEpisodeBuffer() for _ in range(self.num_agents)] for _ in range(self.num_slaves)] for _ in range(self.num_h)]
		self.tutorial_episodes = [[[RNNEpisodeBuffer() for _ in range(self.num_agents)] for _ in range(self.num_slaves)] for _ in range(self.num_h)]
        
		self.indexToNetDic = {0:0, 1:0}

		self.filecount = 0

		self.num_td_reset = [0, 0]

		self.num_td_reset_at_goal = [0, 0]

		self.env.slaveResets()


	def loadTargetModels(self,path,index):
		for h in range(self.num_h):
			self.target_model[h][self.indexToNetDic[index]].load(nndir+'/'+path+'_'+str(self.indexToNetDic[index%self.num_agents])+'_'+str(h)+'.pt')
		if os.path.isfile(nndir+'/rms.ms'):
			self.rms.load(nndir+'/rms.ms')

	def saveModels(self):
		for i in range(self.num_policy):
			for h in range(self.num_h):
				self.target_model[h][i].save(nndir+'/'+'current_'+str(i)+'_'+str(h)+'.pt')

			if self.max_return_epoch == self.num_evaluation:
				for h in range(self.num_h):
					self.target_model[h][i].save(nndir+'/'+'max_'+str(i)+'_'+str(h)+'.pt')

			if self.num_evaluation%100 == 0:
				for h in range(self.num_h):
					self.target_model[h][i].save(nndir+'/'+str(self.num_evaluation)+'_'+str(i)+'_'+str(h)+'.pt')

		self.rms.save(nndir+'/rms.ms')


	def arrayToOneHotVector(nparr):
		result = np.array(list(np.copy(nparr)))
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
		self.sum_return = 0;
		self.num_episode = 0;
		self.num_correct_throwing = 0;
		self.num_td_reset = [0, 0]
		self.num_td_reset_at_goal = [0, 0]
		for h in range(self.num_h):
			for index in range(self.num_policy):
				self.num_tuple[h][index] = 0
				self.num_tutorial_tuple[h][index] = 0

		self.total_episodes = [ [[] for i in range(self.num_policy)] for _ in range(self.num_h)]
		self.total_tutorial_episodes = [ [[] for i in range(self.num_policy)] for _ in range(self.num_h)]

		states = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]
		actions = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]
		rewards = [[None for _ in range(self.num_slaves)] for _ in range(self.num_agents)]
		accRewards = [[0 for _ in range(self.num_slaves)] for _ in range(self.num_agents)]

		states_h = np.array([None for _ in range(self.num_h)])
		actions_h = np.array([[None for _ in range(self.num_agents)] for _ in range(self.num_h)])
		logprobs_h = np.array([[None for _ in range(self.num_agents)] for _ in range(self.num_h)])
		values_h = np.array([[None for _ in range(self.num_agents)] for _ in range(self.num_h)])
		is_exploitation = np.array([None for _ in range(self.num_agents)])
		
		followTutorial = [False]*self.num_slaves

		onFoulResetProcess = [False]*self.num_slaves

		terminated = [False]*self.num_slaves*self.num_agents

		for i in range(self.num_agents):
			for j in range(self.num_slaves):
				states[i][j] = self.env.getState(j,i).astype(np.float32)

		states = np.array(states)
		states[:,:,:-5] = self.rms.apply(states[:,:,:-5])
		mask = states[:,:,-5:]

		learningTeam = 0
		teamDic = {0: 0, 1: 0}

		local_step = 0
		counter = 0

		def arrayToOneHotVectorWithConstraint(nparr):
			result = np.array(list(np.copy(nparr)))

			for agent in range(len(nparr)):
				for slaves in range(len(nparr[agent])):
					maxIndex = 0
					maxValue = -100
					for i in range(len(nparr[agent][slaves])):
						result[agent][slaves][i] = 0.0
						if nparr[agent][slaves][i] > maxValue:
							maxValue = nparr[agent][slaves][i]
							maxIndex = i
					maxIndex = self.env.setActionType(maxIndex, slaves, agent)
					result[agent][slaves][maxIndex] = 1.0
			return result

		def arrayToScalarVectorWithConstraint(nparr, isNew):
			temp = np.array(list(np.copy(nparr)))
			tempShape = np.shape(temp)
			scalarVectorShape = list(tempShape)
			scalarVectorShape[2] = 1

			result = np.zeros(scalarVectorShape);

			result_oneHot = np.array(list(np.copy(nparr)))

			oneHotVectorShape = list(tempShape)
			oneHotVectorShape[2] = self.num_action_types

			result_oneHot = np.zeros(oneHotVectorShape)

			for agent in range(len(nparr)):
				for slaves in range(len(nparr[agent])):
					maxIndex = 0
					maxValue = -100
					for i in range(self.num_action_types):
						result_oneHot[agent][slaves][i] = 0.0
						if nparr[agent][slaves][i] > maxValue:
							maxValue = nparr[agent][slaves][i]
							maxIndex = i
					maxIndex = self.env.setActionType(maxIndex, slaves, agent, isNew)
					result[agent][slaves] = maxIndex
					result_oneHot[agent][slaves][maxIndex] = 1.0

			return result, result_oneHot


		def getActionTypeFromVector(vec):
			maxIndex = 0
			maxValue = -100
			for i in range(len(vec)):
				if vec[i] == 1:
					return i
			print("Something wrong in vec")
			print(vec)
			return 0

		h_slave = [None]*self.num_agents

		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')

			tutorialRatio = 0.0

			useEmbeding = True
			# generate transition of first hierachy

			a_dist_slave = [None]*self.num_agents
			v_slave = [None]*self.num_agents

			if counter%self.typeFreq == 1:
				states_h[0] = np.copy(states)

				for i in range(self.num_agents):
					if teamDic[i] == learningTeam:
						a_dist_slave_agent,v_slave_agent = \
							self.target_model[0][self.indexToNetDic[i]].forward(Tensor(states_h[0][i]))
						a_dist_slave[i] = a_dist_slave_agent
						v_slave[i] = v_slave_agent
						actions_h[0][i] = a_dist_slave[i].sample().cpu().detach().numpy().squeeze().squeeze();
						for j in range(self.num_slaves):
							if followTutorial[j] is True:
								actions_h[0][i][j][0:2] = self.env.getCorrectActionType(j,i)

				for i in range(self.num_agents):
					if teamDic[i] == learningTeam:
						logprobs_h[0][i] = a_dist_slave[i].log_prob(Tensor(actions_h[0][i]))\
							.cpu().detach().numpy().reshape(-1);
						values_h[0][i] = v_slave[i].cpu().detach().numpy().reshape(-1);

				for i in range(self.num_agents):
					for j in range(self.num_slaves):
						if self.env.isOnFoulReset(j):
							temp_a_dist,_ = self.target_model[0][self.indexToNetDic[i]].forward(\
								Tensor([states_h[0][i][j]]),self.stdScale)
							actions_h[0][i][j] = temp_a_dist.sample().cpu().detach().numpy()
							logprobs_h[0][i][j] = temp_a_dist.log_prob(Tensor(actions_h[0][i][j])).cpu().detach().numpy().reshape(-1)[0]
	

			actions_c = np.array(list(np.copy(actions_h[0])))

			actions_c = actions_c[:,:,-self.num_c:]

			actions_0_scalar = None
			actions_0_oneHot = None
			useEmbeding = False

			if counter%self.typeFreq == 1:
				actions_0_scalar, actions_0_oneHot = arrayToScalarVectorWithConstraint(actions_h[0], True)
			else:
				actions_0_scalar, actions_0_oneHot = arrayToScalarVectorWithConstraint(actions_h[0], False)

			action_embeding_ones = np.ones(np.shape(states_h[0]),dtype=np.float32)

			if useEmbeding:
				actions_0_oneHot = actions_0_oneHot*0
				action_embeding_ones = 0.5 * action_embeding_ones*actions_0_scalar
			else:
				action_embeding_ones = 0.0 * action_embeding_ones*actions_0_scalar

			# generate transition of second hierachy
			for h in range(1,self.num_h):
				if h == 1:
					embededState = states+action_embeding_ones
					states_h[h] = np.concatenate((embededState, actions_0_oneHot), axis=2)

				a_dist_slave = [None]*self.num_agents
				v_slave = [None]*self.num_agents
				for i in range(self.num_agents):
					if teamDic[i] == learningTeam:
						a_dist_slave_agent,v_slave_agent = self.target_model[h][self.indexToNetDic[i]].forward(\
							Tensor([states_h[h][i]]))
						a_dist_slave[i] = a_dist_slave_agent
						v_slave[i] = v_slave_agent
						actions_h[h][i] = a_dist_slave[i].sample().cpu().detach().numpy().squeeze().squeeze();

						for j in range(self.num_slaves):
							if followTutorial[j] is True:
								print("line628")
								exit(0)
								decodedActionDetail = self.env.getCorrectActionDetail(j,i)
								encodedActionDetail, _ = self.actionEncoder.encode(Tensor(decodedActionDetail))
								encodedActionDetail= encodedActionDetail.cpu().detach().numpy()
								actions_h[h][i][j] = encodedActionDetail


				for i in range(self.num_agents):
					if teamDic[i] == learningTeam:
						logprobs_h[h][i] = a_dist_slave[i].log_prob(Tensor(actions_h[h][i]))\
							.cpu().detach().numpy().reshape(-1);
						values_h[h][i] = v_slave[i].cpu().detach().numpy().reshape(-1);

				for i in range(self.num_agents):
					for j in range(self.num_slaves):
						if self.env.isOnFoulReset(j):
							temp_a_dist,_ = self.target_model[h][self.indexToNetDic[i]].forward(\
								Tensor([states_h[h][i][j]]),self.stdScale)
							actions_h[h][i][j] = temp_a_dist.sample().cpu().detach().numpy()
							logprobs_h[h][i][j] = temp_a_dist.log_prob(Tensor(actions_h[h][i][j])).cpu().detach().numpy().reshape(-1)[0]


			actionsDecodePart = np.array(list(actions_h[h]))

			decodeShape = list(np.shape(actionsDecodePart))
			decodeShape[2] = 9
			actionsDecoded =np.empty(decodeShape,dtype=np.float32)

			for i in range(len(actionsDecodePart)):
				for j in range(len(actionsDecodePart[i])):
					curActionType = int(actions_0_scalar[i][j][0])
					actionsDecoded[i][j] = self.actionDecoders[curActionType].decode(Tensor(actionsDecodePart[i][j])).cpu().detach().numpy()

			envActions = actionsDecoded

			for i in range(self.num_agents):
				for j in range(self.num_slaves):

					self.env.setAction(envActions[i][j], j, i);

			self.env.stepsAtOnce()

			nan_occur = [False]*self.num_slaves

			for j in range(self.num_slaves):
				if not self.env.isOnResetProcess(j):
					for i in range(self.num_agents):
						if teamDic[i] == learningTeam:
							rewards[i][j] = self.env.getReward(j, i, True)
							if counter%self.typeFreq == 1:
								accRewards[i][j] = rewards[i][j]
							else:
								accRewards[i][j] += rewards[i][j]

							if np.any(np.isnan(rewards[i][j])):
								nan_occur[j] = True
							if np.any(np.isnan(states[i][j])) or np.any(np.isnan(envActions[i][j])):
								nan_occur[j] = True

							if followTutorial[j] is False:
								if not onFoulResetProcess[j]:
									self.sum_return += rewards[i][j]

							if accRewards[i][j] > 0.0 and not self.env.isTerminalState(j):
								embed()
								exit(0)

			for j in range(self.num_slaves):
				if not self.env.isOnResetProcess(j):
					if nan_occur[j] is True:
						for i in range(self.num_agents):
							if teamDic[i] == learningTeam:
								for h in range(self.num_h):
									if followTutorial[j] is False:
										self.total_episodes[h][self.indexToNetDic[i]].append(self.episodes[h][j][i])
										self.episodes[h][j][i] = RNNEpisodeBuffer()
									else:
										self.total_episodes[h][self.indexToNetDic[i]].append(self.tutorial_episodes[h][j][i])
										self.tutorial_episodes[h][j][i] = RNNEpisodeBuffer()
								if followTutorial[j] is False:
									self.num_episode += 1

						print("nan", file=sys.stderr)
						self.env.slaveReset(j)
						self.env.setResetCount(self.resetDuration-counter%self.typeFreq, j);
						followTutorial[j] = random.random()<tutorialRatio



					# here not considered followtutorial
					if not onFoulResetProcess[j]:
						for i in range(self.num_agents):
							if len(self.episodes[0][j][i].data) > 0:
								for h in range(self.num_h):
									if h == 0:
										# if (self.env.isTerminalState(j) and not self.env.isTimeOut(j)) or counter%self.typeFreq == 0:
										if (self.env.isTerminalState(j) and not self.env.isTimeOut(j)):
											savedFrameDiff = self.env.getSavedFrameDiff(j)
											if savedFrameDiff <= 10:
												break;
											TDError = self.episodes[h][j][i].getLastData().value -\
											 (self.episodes[h][j][i].getLastData().r + self.gamma*values_h[h][i][j])
											if self.env.isTerminalState(j) :
												TDError = self.episodes[h][j][i].getLastData().value -\
												 (self.episodes[h][j][i].getLastData().r + self.gamma*accRewards[i][j])
											TDError = abs(TDError)
											TDError = 2.0*pow(TDError, 1.0)
											TDError = min(TDError, 0.9)
											if random.random()<TDError : 
												self.env.setToFoulState(j)
												onFoulResetProcess[j] = True;
												for h_ in range(self.num_h):
													if h_ == 0:
														if followTutorial[j] is False:
															self.episodes[h_][j][i].push(states_h[h_][i][j], actions_h[h_][i][j],\
															accRewards[i][j], values_h[h_][i][j], logprobs_h[h_][i][j])
															self.num_tuple[h_][self.indexToNetDic[i]] += 1
															self.total_episodes[h_][self.indexToNetDic[i]].append(self.episodes[h_][j][i])

															if accRewards[i][j] >= 1.0:
																self.num_correct_throwing += 1

															lastTuple = self.episodes[h_][j][i].data[-(1+int(savedFrameDiff/self.typeFreq))]

															self.episodes[h_][j][i] = RNNEpisodeBuffer()
															self.episodes[h_][j][i].data.append(lastTuple)

													else:
														if followTutorial[j] is False:
															self.episodes[h_][j][i].push(states_h[h_][i][j], actions_h[h_][i][j],\
																rewards[i][j], values_h[h_][i][j], logprobs_h[h_][i][j])
															self.num_tuple[h_][self.indexToNetDic[i]] += 1
															self.total_episodes[h_][self.indexToNetDic[i]].append(self.episodes[h_][j][i])

															lastTuple = self.episodes[h_][j][i].data[-(1+savedFrameDiff)]

															self.episodes[h_][j][i] = RNNEpisodeBuffer()
															self.episodes[h_][j][i].data.append(lastTuple)


												self.num_td_reset[h] += 1
												if self.env.isTerminalState(j) : 
													self.num_td_reset_at_goal[h] += 1

												if followTutorial[j] is False:
													self.num_episode += 1

												break
									else:
										# continue
										# if (self.env.isTerminalState(j) and not self.env.isTimeOut(j)) or True:
										if (self.env.isTerminalState(j) and not self.env.isTimeOut(j)):
											savedFrameDiff = self.env.getSavedFrameDiff(j)
											if savedFrameDiff <= 10:
												break;

											TDError = self.episodes[h][j][i].getLastData().value -\
											 (self.episodes[h][j][i].getLastData().r + self.gamma*values_h[h][i][j])
											if self.env.isTerminalState(j) :
												TDError = self.episodes[h][j][i].getLastData().value -\
												 (self.episodes[h][j][i].getLastData().r + self.gamma*rewards[i][j])
											TDError = abs(TDError)
											TDError = 2.0*pow(TDError, 1.0)
											TDError = min(TDError, 0.9)
											if random.random()<TDError :
												self.env.setToFoulState(j)
												onFoulResetProcess[j] = True;
												for h_ in range(self.num_h):
													if h_ == 0:
														if followTutorial[j] is False:
															self.episodes[h_][j][i].push(states_h[h_][i][j], actions_h[h_][i][j],\
															accRewards[i][j], values_h[h_][i][j], logprobs_h[h_][i][j])
															self.num_tuple[h_][self.indexToNetDic[i]] += 1
															self.total_episodes[h_][self.indexToNetDic[i]].append(self.episodes[h_][j][i])

															
															if accRewards[i][j] >= 1.0:
																self.num_correct_throwing += 1

															lastTuple = self.episodes[h_][j][i].data[-(1+int(savedFrameDiff/self.typeFreq))]

															self.episodes[h_][j][i] = RNNEpisodeBuffer()
															self.episodes[h_][j][i].data.append(lastTuple)

													else :
														if followTutorial[j] is False:
															self.episodes[h_][j][i].push(states_h[h_][i][j], actions_h[h_][i][j],\
																	rewards[i][j], values_h[h_][i][j], logprobs_h[h_][i][j])
															self.num_tuple[h_][self.indexToNetDic[i]] += 1
															self.total_episodes[h_][self.indexToNetDic[i]].append(self.episodes[h_][j][i])

															lastTuple = self.episodes[h_][j][i].data[-(1+savedFrameDiff)]

															self.episodes[h_][j][i] = RNNEpisodeBuffer()
															self.episodes[h_][j][i].data.append(lastTuple)


												self.num_td_reset[h] += 1

												if followTutorial[j] is False:
													self.num_episode += 1

												if self.env.isTerminalState(j) : 
													self.num_td_reset_at_goal[h] += 1

					if onFoulResetProcess[j] is True:
						if counter%self.typeFreq != 1:
							continue

					if self.env.isFoulState(j) is True:
						for i in range(self.num_agents):
							if teamDic[i] == learningTeam:
								if onFoulResetProcess[j] is True:
									onFoulResetProcess[j] = False
									accRewards[i][j] = 0.0

								if followTutorial[j] is False:
									while len(self.episodes[0][j][i].data)>(120/self.typeFreq * 4) :
										self.episodes[0][j][i].popleft()
									while len(self.episodes[1][j][i].data)>121 :
										self.episodes[1][j][i].popleft()

									self.num_tuple[0][self.indexToNetDic[i]] += len(self.episodes[0][j][i].data)
									self.num_tuple[1][self.indexToNetDic[i]] += len(self.episodes[1][j][i].data)
									local_step +=  len(self.episodes[1][j][i].data)

								local_step += 1
							actions_h[0][i][j] = self.episodes[0][j][i].getLastData().a

						self.env.foulReset(j)
						followTutorial[j] = random.random()<tutorialRatio

					elif self.env.isTerminalState(j) is True:
						for i in range(self.num_agents):
							if teamDic[i] == learningTeam:
								for h in range(self.num_h):
									if h == 0:
										if followTutorial[j] is False:
											if self.env.isOnFoulReset(j):
												temp_s = self.episodes[h][j][i].getLastData().s
												temp_a = self.episodes[h][j][i].getLastData().a
												temp_r = accRewards[i][j]
												temp_value = self.episodes[h][j][i].getLastData().value
												temp_logprob = self.episodes[h][j][i].getLastData().logprob
												self.episodes[h][j][i].pop()
												self.episodes[h][j][i].push(temp_s, temp_a, temp_r, temp_value, temp_logprob)
											else:
												self.episodes[h][j][i].push(states_h[h][i][j], actions_h[h][i][j],\
												accRewards[i][j], values_h[h][i][j], logprobs_h[h][i][j])
												self.num_tuple[h][self.indexToNetDic[i]] += 1
											if accRewards[i][j] > 0.0:
												self.num_correct_throwing += 1

									else :
										if followTutorial[j] is False:
											self.episodes[h][j][i].push(states_h[h][i][j], actions_h[h][i][j],\
												rewards[i][j], values_h[h][i][j], logprobs_h[h][i][j])
											self.num_tuple[h][self.indexToNetDic[i]] += 1


								for h in range(self.num_h):
									self.total_episodes[h][self.indexToNetDic[i]].append(self.episodes[h][j][i])
									self.episodes[h][j][i] = RNNEpisodeBuffer()

								if followTutorial[j] is False:
									self.num_episode += 1
								local_step += 1
						self.env.slaveReset(j)
						followTutorial[j] = random.random()<tutorialRatio
						self.env.setResetCount(self.resetDuration-counter%self.typeFreq, j);

					else:
						for i in range(self.num_agents):
							if teamDic[i] == learningTeam:
								for h in range(self.num_h):
									if h == 0:
										if counter%self.typeFreq == 0:
											if followTutorial[j] is False:
												if self.env.isOnFoulReset(j):
													# print("111")
													temp_s = self.episodes[h][j][i].getLastData().s
													temp_a = self.episodes[h][j][i].getLastData().a
													temp_r = accRewards[i][j]
													temp_value = self.episodes[h][j][i].getLastData().value
													temp_logprob = self.episodes[h][j][i].getLastData().logprob
													self.episodes[h][j][i].pop()
													self.episodes[h][j][i].push(temp_s, temp_a, temp_r, temp_value, temp_logprob)
												else:
													self.episodes[h][j][i].push(states_h[h][i][j], actions_h[h][i][j],\
													accRewards[i][j], values_h[h][i][j], logprobs_h[h][i][j])
													self.num_tuple[h][self.indexToNetDic[i]] += 1												
									else :
										if followTutorial[j] is False:
											self.episodes[h][j][i].push(states_h[h][i][j], actions_h[h][i][j],\
												rewards[i][j], values_h[h][i][j], logprobs_h[h][i][j])
											self.num_tuple[h][self.indexToNetDic[i]] += 1
								local_step += 1
					

			if local_step >= self.buffer_size:
				for j in range(self.num_slaves):
					for i in range(self.num_agents):
						if teamDic[i] == learningTeam:
							for h in range(self.num_h):
								self.total_episodes[h][self.indexToNetDic[i]].append(self.episodes[h][j][i])
								self.episodes[h][j][i] = RNNEpisodeBuffer()

							if followTutorial[j] is False:
								self.num_episode += 1

					self.env.slaveReset(j)
					followTutorial[j] = random.random()<tutorialRatio
				break

			for i in range(self.num_agents):
				for j in range(self.num_slaves):
					states[i][j] = self.env.getState(j,i).astype(np.float32)
			states = np.array(states)
			states[:,:,:-5] = self.rms.apply(states[:,:,:-5])

		print('SIM : {}'.format(local_step))


	def computeTDandGAE(self):
		for h in range(self.num_h):
			for index in range(self.num_policy):
				self.buffer[h][index].clear()

				for epi in self.total_episodes[h][index]:
					data = epi.getData()
					size = len(data)
					if size == 0:
						continue
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

					if not np.isnan(epi_return):

						TD = values[:size] + advantages


						rnn_replay_buffer = RNNReplayBuffer(10000)
						for i in range(size):

							rnn_replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

						self.buffer[h][index].push(rnn_replay_buffer)


		''' counting numbers '''
		for h in range(self.num_h):
			for index in range(self.num_policy):
				self.num_tuple_so_far[h][index] += self.num_tuple[h][index]

	def tutorial_computeTDandGAE(self):

		for h in range(self.num_h):
			for index in range(self.num_policy):
				self.tutorial_buffer[h][index].clear()
				for epi in self.total_tutorial_episodes[h][index]:
					data = epi.getData()
					size = len(data)
					if size == 0:
						continue
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

					if not np.isnan(epi_return):
						TD = values[:size] + advantages
						rnn_replay_buffer = RNNReplayBuffer(10000)
						for i in range(size):

							rnn_replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

						self.tutorial_buffer[h][index].push(rnn_replay_buffer)




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

				batch_size_h = self.batch_size
				if h == 0:
					 batch_size_h = self.batch_size//8
				np.random.shuffle(all_segmented_transitions)
				for i in range(len(all_segmented_transitions)//batch_size_h):
					batch_segmented_transitions = all_segmented_transitions[i*batch_size_h:(i+1)*batch_size_h]

					batch = RNNTransition(*zip(*batch_segmented_transitions))

					stack_s = np.vstack(batch.s).astype(np.float32)
					stack_a = np.vstack(batch.a).astype(np.float32)
					stack_lp = np.vstack(batch.logprob).astype(np.float32)
					stack_td = np.vstack(batch.TD).astype(np.float32)
					stack_gae = np.vstack(batch.GAE).astype(np.float32)

					a_dist,v = self.target_model[h][buff_index].forward(Tensor(stack_s))	
					
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
					self.optimizer[h][buff_index].zero_grad()

					loss.backward()

					detectNan = False
					for param in self.target_model[h][buff_index].parameters():
						if torch.isnan(param.grad).any():
							detectNan = True
						if param.grad is not None:
							param.grad.data.clamp_(-0.5, 0.5)

					if not detectNan:
						self.optimizer[h][buff_index].step()
					self.sum_loss_actor[h][buff_index] += loss_actor.detach()*batch_size_h/self.num_epochs
					self.sum_loss_critic[h][buff_index] += loss_critic.detach()*batch_size_h/self.num_epochs

					loss_entropy = loss_entropy.detach()
					loss_actor = loss_actor.detach()
					loss_critic = loss_critic.detach()
					loss= loss.detach()

				print('Optimizing actor-critic nn_{} : {}/{}'.format(h, j+1,self.num_epochs),end='\r')
			print('')
		# print('')


	def tutorial_optimizeNN_h(self, h = 0):
		for i in range(self.num_policy):
			self.sum_tutorial_loss_critic[h][i] = 0.0

		for buff_index in range(self.num_policy):
			all_rnn_replay_buffer= np.array(self.tutorial_buffer[h][buff_index].buffer)
			for j in range(self.num_epochs):
				all_segmented_transitions = []
				for rnn_replay_buffer in all_rnn_replay_buffer:
					rnn_replay_buffer_size = len(rnn_replay_buffer.buffer)
					for i in range(rnn_replay_buffer_size):
						all_segmented_transitions.append(rnn_replay_buffer.buffer[i])

				np.random.shuffle(all_segmented_transitions)

				tutorial_batchSize = self.batch_size//16
				for i in range(len(all_segmented_transitions)//tutorial_batchSize):
					batch_segmented_transitions = all_segmented_transitions[i*tutorial_batchSize:(i+1)*tutorial_batchSize]

					batch = RNNTransition(*zip(*batch_segmented_transitions))

					stack_s = np.vstack(batch.s).astype(np.float32)
					stack_a = np.vstack(batch.a).astype(np.float32)
					stack_lp = np.vstack(batch.logprob).astype(np.float32)
					stack_td = np.vstack(batch.TD).astype(np.float32)
					stack_gae = np.vstack(batch.GAE).astype(np.float32)

					a_dist,v = self.target_model[h][buff_index].forward(Tensor(stack_s))	

					loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()

					loss = loss_critic
					self.optimizer[h][buff_index].zero_grad()

					loss.backward()

					detectNan = False
					for param in self.target_model[h][buff_index].parameters():
						if torch.isnan(param.grad).any():
							detectNan = True
							# print("nan")
						if param.grad is not None:
							param.grad.data.clamp_(-0.5, 0.5)

					self.sum_tutorial_loss_critic[h][buff_index] += loss_critic.detach()*tutorial_batchSize/self.num_epochs
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

		self.generateTransitions()
		self.computeTDandGAE()
		self.optimizeModel()


	def optimizeModel(self):
		for h in range(self.num_h):
			self.optimizeNN_h(h)


	def evaluate(self):
		self.num_evaluation = self.num_evaluation + 1
		hour = int((time.time() - self.tic)//3600.0)
		m = int((time.time() - self.tic)//60.0)
		s = int((time.time() - self.tic))
		m = m - hour*60
		s = int((time.time() - self.tic))
		s = s - hour*3600 - m*60
		if self.num_episode is 0:
			self.num_episode = 1

		for h in range(self.num_h):
			for i in range(self.num_policy):
				if self.num_tuple[h][i] is 0:
					self.num_tuple[h][i] = 1
				if self.num_tutorial_tuple[h][i] is 0:
					self.num_tutorial_tuple[h][i] = 1

		if self.max_return < self.sum_return/self.num_episode:
			self.max_return = self.sum_return/self.num_episode
			self.max_return_epoch = self.num_evaluation

		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,hour,m,s))
		print('||--------------ActorCriticNN------------------')

		for i in range(self.num_policy):
			for h in range(self.num_h):
				print('||Avg Loss Actor {} {}         : {:.4f}'.format(h, i, self.sum_loss_actor[h][i]/self.num_tuple[h][i]))
				print('||Avg Loss Critic {} {}        : {:.4f}'.format(h, i, self.sum_loss_critic[h][i]/self.num_tuple[h][i]))
				print('||Noise {}                     : {:.3f}'.format(h, self.target_model[h][i].log_std.exp().mean()))		


				print('||Num Transition So far {} {}  : {}'.format(h, i, self.num_tuple_so_far[h][i]))
				print('||Num Transition {} {}         : {}'.format(h, i, self.num_tuple[h][i]))


		print('||Num Episode              : {}'.format(self.num_episode))
		for i in range(self.num_policy):
			print('||Avg Step per episode {}   : {:.1f}'.format(i, self.num_tuple[1][i]/self.num_episode))

		print('||Avg Num Correct Throwings per episode   : {:.3f}'.format(self.num_correct_throwing/self.num_episode))
		print('||Max Return per episode   : {:.3f}'.format(self.max_return))
		print('||Avg Return per episode   : {:.3f}'.format(self.sum_return/self.num_episode))

		for h in range(self.num_h):
			print('||Num TD Reset             : {} {}'.format(h, self.num_td_reset[h]))
			print('||Num TD Reset at goal     : {} {}'.format(h, self.num_td_reset_at_goal[h]))

		self.rewards.append(self.sum_return/self.num_episode)
		self.numSteps.append(self.num_tuple[1][0]/self.num_episode)
		self.num_correct_throwings.append(self.num_correct_throwing/self.num_episode)
		self.saveModels()
		

		summary.add_scalars('rewards and num_correct_throwings',{'rewards' : self.sum_return/self.num_episode, 'num_ct' : self.num_correct_throwing/self.num_episode}, self.num_evaluation)
		summary.add_scalar('numSteps',self.num_tuple[1][0]/self.num_episode, self.num_evaluation)

		for name, param in self.target_model[0][0].named_parameters():
			summary.add_histogram(name+"_actionType", param.clone().cpu().data.numpy(), self.num_evaluation)
		for name, param in self.target_model[1][0].named_parameters():
			summary.add_histogram(name+"_actionDetail", param.clone().cpu().data.numpy(), self.num_evaluation)
		
		print('=============================================')
		return np.array(self.rewards), np.array(self.numSteps), np.array(self.num_correct_throwings)





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



def plot_numSteps(y,title,num_fig=1,ylim=True,path=""):
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
	if ylim:
		plt.ylim([0,1])

	fig.canvas.draw()
	fig.canvas.flush_events()

import argparse
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')
	# parser.add_argument('-p','--policy',help='pretrained pollicy path')
	parser.add_argument('-iter','--iteration',help='num iterations')
	parser.add_argument('-n','--name',help='name of training setting')
	parser.add_argument('-motion', '--motion', help='motion nn path')

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

	result_figure = nndir+"/"+"result.png"
	result_figure_numSteps = nndir+"/"+"result_numSteps.png"
	result_figure_numCTs = nndir+"/"+"result_numCTs.png"
	result_figure_num = 0
	while Path(result_figure).is_file():
		result_figure = nndir+"/"+"result_{}.png".format(result_figure_num)
		result_figure_numSteps = nndir+"/"+"result_numSteps_{}.png".format(result_figure_num)
		result_figure_numCTs = nndir+"/"+"result_numCTs_{}.png".format(result_figure_num)
		result_figure_num+=1

	for i in range(5000000):
		rl.train()
		rewards, numSteps, numCTs = rl.evaluate()
