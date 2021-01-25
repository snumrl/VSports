import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils.dataLoader import loadData
from IPython import embed

import os
from os.path import join, exists
from os import mkdir

import sys
import random
import numpy as np
from pathlib import Path

from VAE import VAE, VAEEncoder, VAEDecoder, loss_function

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-3)
# print("asdf")
nnCount = 0
vaeDir = "vae_nn_sep_"+str(nnCount)

if not exists(vaeDir):
    mkdir(vaeDir)

def getActionTypeFromVector(actionTypeVector):
	maxValue = -100
	maxValueIndex = 0
	# print(actionTypeVector)

	for i in range(len(actionTypeVector)):

		if actionTypeVector[i] > maxValue:
			maxValue = actionTypeVector[i]
			maxValueIndex = i
	return maxValueIndex

# def splitControlVector(cv_list, numActionTypes):
# 	# embed()
# 	# exit(0)
# 	actionTypeVector_list = cv_list[:,4:4+numActionTypes]
# 	# controlVectorLists = [[] for _ in range(numActionTypes)]
# 	controlVectorLists = np.empty((numActionTypes, 1, 14), dtype=np.float32) 
# 	# concated = []
# 	for i in range(int(len(cv_list))):
# 		curActionTypeVector = actionTypeVector_list[i]
# 		curActionType = getActionTypeFromVector(curActionTypeVector)
# 		# print(curActionType)
# 		# controlVectorLists[curActionType].append(cv_list[i])

# 		# embed()
# 		# exit(0)
# 		if i == 0:
# 			controlVectorLists[curActionType][0] = np.array([cv_list[i]])
# 		else:
# 			controlVectorLists[curActionType] = np.concatenate((controlVectorLists[curActionType], np.array([cv_list[i]])),axis = 0)

# 	return controlVectorLists



def splitControlVector(cv_list, numActionTypes):
	# embed()
	# exit(0)
	actionTypeVector_list = cv_list[:,4:4+numActionTypes]
	controlVectorLists = [[] for _ in range(numActionTypes)]
	# controlVectorLists = np.empty((numActionTypes, 1, 14), dtype=np.float32) 
	# concated = []
	for i in range(int(len(cv_list))):
		curActionTypeVector = actionTypeVector_list[i]
		curActionType = getActionTypeFromVector(curActionTypeVector)
		# print(curActionType)
		controlVectorLists[curActionType].append(cv_list[i])

		# embed()
		# exit(0)
		# if i == 0:
		# 	controlVectorLists[curActionType][0] = np.array([cv_list[i]])
		# else:
		# 	controlVectorLists[curActionType] = np.concatenate((controlVectorLists[curActionType], np.array([cv_list[i]])),axis = 0)

	for i in range(numActionTypes):
		controlVectorLists[i] = np.array(controlVectorLists[i])
	return np.array(controlVectorLists)


def removeActionTypePart(splited_cv_list, numActionTypes):
	removed_cv_list = []
	for i in range(numActionTypes):
		# embed()
		# exit(0)
		front_list = splited_cv_list[i][:,0:4]
		end_list = splited_cv_list[i][:,4+numActionTypes:14]
		# embed()
		# exit(0)
		removed_list = np.concatenate((front_list,end_list), axis= 1)
		removed_cv_list.append(removed_list)
	return np.array(removed_cv_list)


class ComprehensiveControlVectorTraining():
	def __init__(self, path, numActionTypes):
		self.path = path
		self.numActionTypes = numActionTypes

		self.VAEEncoder = VAEEncoder().to(device)
		self.VAEDecoders = [VAEDecoder().to(device) for _ in range(self.numActionTypes)]
		self.optimizers = []
		for i in range(self.numActionTypes):
			self.optimizers.append(optim.Adam(list(self.VAEEncoder.parameters()) + list(self.VAEDecoders[i].parameters()), lr=1e-5))

		self.controlVectorList = loadData(os.path.dirname(os.path.abspath(__file__)) + "/../extern/ICA/motions/%s/data/0_xData.dat"%(path))
		self.controlVectorList = np.array(self.controlVectorList)
		self.controlVectorListSplited = splitControlVector(self.controlVectorList, numActionTypes)
		# self.controlVectorListSplited = splitControlVector(self.controlVectorList , numActionTypes)
		self.controlVectorListSplited = removeActionTypePart(self.controlVectorListSplited, numActionTypes)

	def trainTargetCV(self,actionType):
		print('Train control vector of actiontype {}, Size : {}'.format(actionType, len(self.controlVectorListSplited[actionType])))
		for epoch in range(3):
			self.VAEEncoder.train()
			# self.VAEDecoders[0].train()
			self.VAEDecoders[actionType].train()
			train_loss = 0
			action_controlVectorList = self.controlVectorListSplited[actionType]
			np.random.shuffle(action_controlVectorList)

			# for i in range(int(len(action_controlVectorList)/64)):
			for i in range(200):
				data = action_controlVectorList[i*64:(i+1)*64]
				data = Tensor(data)
				# self.optimizers[0].zero_grad()
				self.optimizers[actionType].zero_grad()
				latent, mu, logvar = self.VAEEncoder(data)

				# recon_batch = self.VAEDecoders[0](latent)
				recon_batch = self.VAEDecoders[actionType](latent)

				# if i == 0:
				# 	print(data.view(-1,9)[0])
				# 	print(recon_batch[0])
				# 	print(latent[0])
				# 	print("")

				loss = loss_function(recon_batch, data, mu, logvar)
				loss.backward()
				train_loss += loss.item()
				# self.optimizers[0].step()
				self.optimizers[actionType].step()
				if i % 40 == 0:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
		                epoch, i * len(data), 12800,
		                100. * i / (200),
		                loss.item() / len(data)))
			print('===> Epoch: {} Arverage loss: {:.4f}'.format(1, train_loss / 12800))
		# self.VAEDecoders[0].save("vae_nn/vae_action_decoder_"+str(actionType)+".pt")	
		self.VAEDecoders[actionType].save(vaeDir + "/vae_action_decoder_"+str(actionType)+".pt")	
		self.VAEEncoder.save(vaeDir + "/vae_action_encoder.pt")	

	# def trainTargetCV(self,actionType1, actionType2):
	# 	print('Train control vector of actiontype {}, Size : {}'.format(actionType1, len(self.controlVectorListSplited[actionType1])))
	# 	print('Train control vector of actiontype {}, Size : {}'.format(actionType2, len(self.controlVectorListSplited[actionType2])))
	# 	for epoch in range(3):
	# 		self.VAEEncoder.train()
	# 		# self.VAEDecoders[0].train()
	# 		self.VAEDecoders[actionType1].train()
	# 		self.VAEDecoders[actionType2].train()
	# 		train_loss = 0
	# 		# embed()
	# 		# exit(0)
	# 		# action_controlVectorList = self.controlVectorListSplited[actionType2]

	# 		action_controlVectorList=np.concatenate((self.controlVectorListSplited[actionType1], self.controlVectorListSplited[actionType2]),axis=0)
	# 		np.random.shuffle(action_controlVectorList)
	# 		# for i in range(int(len(action_controlVectorList)/64)):
	# 		for i in range(200):
	# 			data = action_controlVectorList[i*64:(i+1)*64]
	# 			data = Tensor(data)
	# 			# self.optimizers[0].zero_grad()
	# 			self.optimizers[actionType1].zero_grad()
	# 			latent, mu, logvar = self.VAEEncoder(data)

	# 			# recon_batch = self.VAEDecoders[0](latent)
	# 			recon_batch = self.VAEDecoders[actionType1](latent)

	# 			# if i == 0:
	# 			# 	print(data.view(-1,9)[0])
	# 			# 	print(recon_batch[0])
	# 			# 	print(latent[0])
	# 			# 	print("")

	# 			loss = loss_function(recon_batch, data, mu, logvar)
	# 			loss.backward()
	# 			train_loss += loss.item()
	# 			# self.optimizers[0].step()
	# 			self.optimizers[actionType1].step()
	# 			if i % 40 == 0:
	# 				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
	# 	                epoch, i * len(data), 12800,
	# 	                100. * i / (200),
	# 	                loss.item() / len(data)))
	# 		print('===> Epoch: {} Arverage loss: {:.4f}'.format(1, train_loss / 12800))
	# 	# self.VAEDecoders[0].save("vae_nn/vae_action_decoder_"+str(actionType)+".pt")	
	# 	self.VAEDecoders[actionType1].save("vae_nn/vae_action_decoder_"+str(actionType1)+".pt")	
	# 	self.VAEDecoders[actionType1].save("vae_nn/vae_action_decoder_"+str(actionType2)+".pt")	
	# 	self.VAEEncoder.save("vae_nn/vae_action_encoder.pt")	

	# def trainComprehensiveLatentSpace(self, actionTypeSource, actionTypeTarget):




def trainControlVector(path, actionType):
	for epoch in range(30):
		model.train()
		train_loss = 0
		control_vector_list = loadData(os.path.dirname(os.path.abspath(__file__)) + "/../extern/ICA/motions/%s/data/0_xData.dat"%(path))
		random.shuffle(control_vector_list)

		# embed()
		# exit(0)
		for i in range(int(len(control_vector_list)/128)):
			data = control_vector_list[i*128:(i+1)*128]
			data = Tensor(data)
			# print(data[0])
			# exit(0)
			# embed()
			# exit(0)
			optimizer.zero_grad()
			recon_batch, mu, logvar = model(data)
			loss = loss_function(recon_batch, data, mu, logvar)
			loss.backward()
			train_loss += loss.item()
			optimizer.step()
			if i % 400 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
	                epoch, i * len(data), len(control_vector_list),
	                100. * i / (len(control_vector_list)/128),
	                loss.item() / len(data)))

		print('===> Epoch: {} Arverage loss: {:.4f}'.format(1, train_loss / len(control_vector_list)))
	model.save(vaeDir + "/vae_0.pt")	

# def test(path):
# 	model.eval()
# 	with torch.no_grad():
# 		control_vector_list = loadData(os.path.dirname(os.path.abspath(__file__)) + "/../extern/ICA/motions/%s/data/0_xData.dat"%(path))
# 		random.shuffle(control_vector_list)
# 		model.load("vae_nn/vae_0.pt")
# 		for i in range(10):
# 			data = control_vector_list[i]
# 			data = Tensor(data)
# 			recon_batch , mu, logvar = model(data)
# 			print(data)
# 			print(recon_batch)
# 			print(mu)
# 			print(logvar)
# 			print("")



# trainControlVector("basket_0")
# test("basket_0")

ccvt = ComprehensiveControlVectorTraining("basket_18", 5)
for i in range(100):
	ccvt.trainTargetCV(0)
	ccvt.trainTargetCV(1)
	ccvt.trainTargetCV(2)
	ccvt.trainTargetCV(3)
	ccvt.trainTargetCV(4)