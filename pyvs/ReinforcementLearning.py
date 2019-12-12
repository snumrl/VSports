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
		self.num_slaves = 8
		self.num_agents = 4
		self.env = Env(self.num_agents)
		self.num_state = self.env.getNumState()
		self.num_action = self.env.getNumAction()

		self.policy = [None]*2
		self.policy[0] = SimulationNN(self.num_state, self.num_action)
		self.policy[1] = SimulationNN(self.num_state, self.num_action)
		

		self.num_epochs = 2
		self.num_evaluation = 0
		self.num_tuple_so_far = 0
		self.num_episode = 0
		self.num_tuple = 0

		self.num_simulation_Hz = self.env.getSimulationHz()
		self.num_control_Hz = self.env.getControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.997
		self.lb = 0.95

		self.buffer_size = 8*1024
		self.batch_size = 256
		# self.trunc_size = 32
		# self.burn_in_size = 8
		# self.bptt_size = 8;

		self.buffer = Buffer(30000)

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
		self.optimizer = [None]*2
		self.optimizer[0] = optim.Adam(self.model[0].parameters(), lr=self.learning_rate)
		self.optimizer[1] = optim.Adam(self.model[1].parameters(), lr=self.learning_rate)

		self.max_iteration = 50000

		self.w_entropy = 0.0001

		self.loss_actor = 0.0
		self.loss_critic = 0.0

		self.rewards = []

		self.sum_return = 0.0

		self.max_return = 0.0

		self.max_return_epoch = 1
		self.tic = time.time()

		self.episodes = [[RNNEpisodeBuffer() for y in range(self.num_agents)] for x in range(self.num_slaves)]

		self.env.resets()

	def loadModel(self,path,index):
		self.model[index].load('../nn/'+path+".pt")

	def loadPolicy(self,path):
		for i in range(2):
			self.policy[i].load(path+"_"+str(i)+".pt")
			self.model[i].ss_policy.load_state_dict(self.policy[i].ss_policy.state_dict())
			# self.model[i].ss_policy = self.policy[i].ss_policy


	def saveModel(self):
		self.model[0].save('../nn/current.pt')

		if self.max_return_epoch == self.num_evaluation:
			self.model[0].save('../nn/max.pt')
		if self.num_evaluation%20 == 0:
			self.model[0].save('../nn/'+str(self.num_evaluation)+'.pt')

	def computeTDandGAE(self):
		# self.total_episodes = self.total_episodes + self.total_hindsight_episodes

		'''Scheduler'''
		self.buffer.clear()
		self.sum_return = 0.0
		for epi in self.total_episodes:
			data = epi.getData()
			size = len(data)
			# print(size)
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
				for i in range(size):
					rnn_replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i], hiddens[i][0], hiddens[i][1])
				self.buffer.push(rnn_replay_buffer)

		''' counting numbers(w.r.t scheduler) '''
		self.num_episode = len(self.total_episodes)
		self.num_tuple = 0
		for rnn_replay_buffer in self.buffer.buffer:
			self.num_tuple += len(rnn_replay_buffer.buffer)
		self.num_tuple_so_far += self.num_tuple



	def getHardcodedAction(self, slave_index, agent_index):
		return np.array([0,0,-1])


	def generateTransitions(self):
		self.total_episodes = []
		self.total_hindsight_episodes = []

		states = [None]*self.num_slaves*self.num_agents
		actions = [None]*self.num_slaves*self.num_agents
		rewards = [None]*self.num_slaves*self.num_agents
		logprobs = [None]*self.num_slaves*self.num_agents
		values = [None]*self.num_slaves*self.num_agents
		'''hiddens : (hidden ,cell) tuple''' 
		hiddens = [None]*self.num_slaves*self.num_agents
		hiddens_forward = [None]*self.num_slaves*self.num_agents
		terminated = [False]*self.num_slaves*self.num_agents

		for i in range(self.num_slaves):
			for j in range(self.num_agents):
				states[i*self.num_agents+j] = self.env.getLocalState(i,j)
				hiddens[i*self.num_agents+j] = self.model[0].init_hidden(1)
				hiddens[i*self.num_agents+j] = (hiddens[i*self.num_agents+j][0].cpu().detach().numpy(), \
							hiddens[i*self.num_agents+j][1].cpu().detach().numpy())
				hiddens_forward[i*self.num_agents+j] = self.model[0].init_hidden(1)


		learningTeam = random.randrange(0,2)
		'''Fixed to team 0'''
		learningTeam = 0
		# teamDic = {0: 0, 1: 1}
		teamDic = {0: 0, 1: 0, 2: 1, 3: 1}

		local_step = 0
		counter = 0

		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')

			useHardCoded = False

			if not useHardCoded:
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
								self.loadModel("0",i*self.num_agents+1)


			''' Scheduler Part '''
			for i in range(self.num_slaves):
				a_dist_slave = []
				v_slave = []
				hiddens_slave = []
				for j in range(self.num_agents):
					if not useHardCoded:
						if teamDic[j] == learningTeam:
							a_dist_slave_agent,v_slave_agent, hiddens_slave_agent = self.model[0].forward(\
								Tensor([states[i*self.num_agents+j]]),(Tensor(hiddens[i*self.num_agents+j][0]), Tensor(hiddens[i*self.num_agents+j][1])))
						else :
							a_dist_slave_agent,v_slave_agent, hiddens_slave_agent = self.model[i*self.num_agents+1].forward(\
								Tensor([states[i*self.num_agents+j]]),(Tensor(hiddens[i*self.num_agents+j][0]), Tensor(hiddens[i*self.num_agents+j][1])))
						a_dist_slave.append(a_dist_slave_agent)
						v_slave.append(v_slave_agent)
						hiddens_slave.append((hiddens_slave_agent[0].cpu().detach().numpy(), hiddens_slave_agent[1].cpu().detach().numpy()))
						actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy()[0][0];		

					else :
						if teamDic[j] == learningTeam:
							a_dist_slave_agent,v_slave_agent, hiddens_slave_agent = self.model[0].forward(\
								Tensor([states[i*self.num_agents+j]]),(Tensor(hiddens[i*self.num_agents+j][0]), Tensor(hiddens[i*self.num_agents+j][1])))
							a_dist_slave.append(a_dist_slave_agent)
							v_slave.append(v_slave_agent)
							
							hiddens_slave.append((hiddens_slave_agent[0].cpu().detach().numpy(), hiddens_slave_agent[1].cpu().detach().numpy()))

							explorationFlag = random.randrange(0,3)
							if explorationFlag == 0:
								actions[i*self.num_agents+j] = a_dist_slave[j].sample().cpu().detach().numpy()[0][0];
							else :
								actions[i*self.num_agents+j] = a_dist_slave[j].loc.cpu().detach().numpy()[0][0];

						else :
							'''dummy'''
							# a_dist_slave_agent,v_slave_agent, hiddens_slave_agent = self.model[0].forward(\
							# 	Tensor([states[i*self.num_agents+j]]),(Tensor(hiddens[i*self.num_agents+j][0]), Tensor(hiddens[i*self.num_agents+j][1])), self.env.getNumIterations())
							# a_dist_slave.append(a_dist_slave_agent)
							# v_slave.append(v_slave_agent)
							# hiddens_slave.append((hiddens_slave_agent[0].cpu().detach().numpy(), hiddens_slave_agent[1].cpu().detach().numpy()))
							actions[i*self.num_agents+j] = self.getHardcodedAction(i, j);

				for j in range(self.num_agents):
					if teamDic[j] == learningTeam:
						logprobs[i*self.num_agents+j] = a_dist_slave[j].log_prob(Tensor(actions[i*self.num_agents+j]))\
							.cpu().detach().numpy().reshape(-1)[0];
						values[i*self.num_agents+j] = v_slave[j].cpu().detach().numpy().reshape(-1)[0];
						hiddens_forward[i*self.num_agents+j] = hiddens_slave[j]


				''' Set the Linear Actor state with scheduler action '''
				for j in range(self.num_agents):
					# if teamDic[j] == learningTeam:

						# self.env.setLinearActorState(i, j, actions[i*self.num_agents+j])
					self.env.setAction(actions[i*self.num_agents+j], i, j);
					# else:

			self.env.stepsAtOnce()


			for i in range(self.num_slaves):
				nan_occur = False
				terminated_state = True
				for k in range(self.num_agents):
					if teamDic[k] == learningTeam:
						rewards[i*self.num_agents+k] = self.env.getReward(i, k) 
						if np.any(np.isnan(rewards[i*self.num_agents+k])):
							nan_occur = True
						if np.any(np.isnan(states[i*self.num_agents+k])) or np.any(np.isnan(actions[i*self.num_agents+k])):
							nan_occur = True

				if nan_occur is True:
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam:
							self.total_episodes.append(self.episodes[i][k])
							self.episodes[i][k] = RNNEpisodeBuffer()

					self.env.reset(i)


				if self.env.isTerminalState(i) is False:
					terminated_state = False
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam:
							self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k],\
								rewards[i*self.num_agents+k], values[i*self.num_agents+k], logprobs[i*self.num_agents+k], hiddens[i*self.num_agents+k])

							local_step += 1

				if terminated_state is True:
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam:
							self.episodes[i][k].push(states[i*self.num_agents+k], actions[i*self.num_agents+k],\
								rewards[i*self.num_agents+k], values[i*self.num_agents+k], logprobs[i*self.num_agents+k], hiddens[i*self.num_agents+k])
							self.total_episodes.append(self.episodes[i][k])
							self.episodes[i][k] = RNNEpisodeBuffer()

					self.env.reset(i)

			if local_step >= self.buffer_size:
				for i in range(self.num_slaves):
					for k in range(self.num_agents):
						if teamDic[k] == learningTeam:
							self.total_episodes.append(self.episodes[i][k])
							self.episodes[i][k] = RNNEpisodeBuffer()

					self.env.reset(i)
				break

			# get the updated state and updated hidden
			for i in range(self.num_slaves):
				for j in range(self.num_agents):
					states[i*self.num_agents+j] = self.env.getLocalState(i,j)
					hiddens[i*self.num_agents+j] = hiddens_forward[i*self.num_agents+j]


		self.env.endOfIteration()
		print('SIM : {}'.format(local_step))

	def optimizeSchedulerNN(self):
		all_rnn_replay_buffer= np.array(self.buffer.buffer)
		for j in range(self.num_epochs):
			all_segmented_transitions = []
			for rnn_replay_buffer in all_rnn_replay_buffer:
				rnn_replay_buffer_size = len(rnn_replay_buffer.buffer)

				for i in range(rnn_replay_buffer_size):
					all_segmented_transitions.append(rnn_replay_buffer.buffer[i])


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

				num_layers =self.model[0].num_layers
				stack_hidden = [Tensor(np.reshape(stack_hidden_h, (num_layers,self.batch_size,-1))), 
									Tensor(np.reshape(stack_hidden_c, (num_layers,self.batch_size,-1)))]


				stack_hidden[0] = stack_hidden[0].detach()
				stack_hidden[1] = stack_hidden[1].detach()


				a_dist,v,cur_stack_hidden = self.model[0].forward(Tensor(stack_s), stack_hidden)	


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

				self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
				self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

				loss = loss_actor + loss_entropy + loss_critic
				self.optimizer.zero_grad()

				# start = time.time()
				loss.backward(retain_graph=True)
				# print("time :", time.time() - start)

				for param in self.model[0].parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5, 0.5)
				self.optimizer.step()

			print('Optimizing scheduler nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')


	def optimizeModel(self):
		self.computeTDandGAE()
		self.optimizeSchedulerNN()


	def train(self):
		frac = 1.0
		self.learning_rate = self.default_learning_rate*frac
		self.clip_ratio = self.default_clip_ratio*frac
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.learning_rate
		self.generateTransitions();
		# self.generateHindsightTransitions();
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
		print('||--------------ActorCriticNN------------------')
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
	slac = SLAC()
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')
	parser.add_argument('-p','--policy',help='pretrained pollicy path')
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

	if args.policy is not None:
		slac.loadPolicy(args.policy)


	else:
		slac.saveModel()
	print('num states: {}, num actions: {}'.format(slac.env.getNumState(),slac.env.getNumAction()))
	# for i in range(ppo.max_iteration-5):
	for i in range(5000000):
		slac.train()
		rewards = slac.evaluate()
		plot(rewards, graph_name + ' scheduler reward',0,False)

