import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils.dataLoader import loadData, loadNormalData
from IPython import embed

import os
from os.path import join, exists
from os import mkdir

import sys
import random
import numpy as np
from pathlib import Path

from CAAE import CAAEEncoder, CAAEDecoder, loss_function
from tensorboardX import SummaryWriter
summary = SummaryWriter()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")

nnCount = 2
vaeDir = "caae_nn"+str(nnCount)

if not exists(vaeDir):
    mkdir(vaeDir)

def getActionTypeFromVector(actionTypeVector):
	maxValue = -100
	maxValueIndex = 0

	for i in range(len(actionTypeVector)):

		if actionTypeVector[i] > maxValue:
			maxValue = actionTypeVector[i]
			maxValueIndex = i
	return maxValueIndex


def splitControlVector(cv_list, numActionTypes):
	# embed()
	# exit(0)
	actionTypeVector_list = cv_list[:,0:numActionTypes]
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
		# front_list = splited_cv_list[i][:,0:4]
		removed_list = splited_cv_list[i][:,numActionTypes:21]
		# embed()
		# exit(0)
		# removed_list = np.concatenate((front_list,end_list), axis= 1)
		removed_cv_list.append(np.array(removed_list))
	# embed()
	# exit(0)
	return np.array(removed_cv_list)

def removeCriticalTimePart(splited_cv_list, numActionTypes):
	removed_cv_list = []
	# for i in range(numActionTypes):
		# embed()
		# exit(0)
		# front_list = splited_cv_list[i][:,0:4]
	removed_list = splited_cv_list[:,0:21]
	# embed()
	# exit(0)
	# removed_list = np.concatenate((front_list,end_list), axis= 1)
	# removed_cv_list.append(np.array(removed_list))
	# embed()
	# exit(0)
	return np.array(removed_list)


class ComprehensiveControlVectorTraining():
	def __init__(self, path, numActionTypes):
		self.path = path
		self.numActionTypes = numActionTypes

		self.CAAEEncoder = CAAEEncoder().to(device)
		self.CAAEDecoder = CAAEDecoder().to(device)
		self.optimizer = optim.Adam(list(self.CAAEEncoder.parameters())+list(self.CAAEDecoder.parameters()), lr=1e-5)

		self.controlVectorList = loadData(os.path.dirname(os.path.abspath(__file__)) + "/../extern/ICA/motions/%s/data/0_xData.dat"%(path))
		self.controlVectorNormal = loadNormalData(os.path.dirname(os.path.abspath(__file__)) + "/../extern/ICA/motions/%s/data/xNormal.dat"%(path))

		self.controlVectorList = np.array(self.controlVectorList)
		self.controlVectorNormal = np.array(self.controlVectorNormal)

		self.actionTypeVectorList = self.controlVectorList * self.controlVectorNormal[1]
		self.actionTypeVectorList += self.controlVectorNormal[0]

		self.controlVectorList[:,:5] = self.actionTypeVectorList[:,:5]

		# self.controlVectorListSplited = splitControlVector(self.controlVectorList, numActionTypes)
		self.controlVectorList = removeCriticalTimePart(self.controlVectorList, numActionTypes)
		# self.controlVectorListSplited = removeActionTypePart(self.controlVectorListSplited, numActionTypes)

	def trainTargetCV(self,num_evaluation):
		# print('Train control vector of actiontype {}, Size : {}'.format(actionType, len(self.controlVectorListSplited[actionType])))
		action_train_loss_BCE = 0
		action_train_loss_KLD = 0
		for epoch in range(4):
			self.CAAEEncoder.train()
			self.CAAEDecoder.train()
			train_loss_BCE = 0
			train_loss_KLD = 0

			action_controlVectorList = self.controlVectorList
			np.random.shuffle(action_controlVectorList)

			for i in range(16):
				data = action_controlVectorList[i*256:(i+1)*256]
				# oneHot_shape = list(np.shape(data))
				# oneHot_shape[1] = self.numActionTypes
				# embed()
				# exit(0)
				oneHot_vec = data[:,0:5]
				data_wo_acitonType = data[:,5:21]

				# combined_data = np.concatenate((oneHot_vec, data_wo_acitonType), axis= 1)

				data_wo_acitonType = Tensor(data_wo_acitonType)
				oneHot_vec = Tensor(oneHot_vec)

				combined_data = torch.cat((oneHot_vec, data_wo_acitonType), axis= 1)

				# embed()
				# exit(0)

				# self.optimizers[0].zero_grad()
				self.optimizer.zero_grad()

				latent, mu, logvar = self.CAAEEncoder(combined_data)

				# recon_batch = self.VAEDecoders[0](latent)
				combined_latent = torch.cat((oneHot_vec, latent), axis= 1)

				recon_batch = self.CAAEDecoder(combined_latent)

				# if i == 0:
				# 	print(data.view(-1,9)[0])
				# 	print(recon_batch[0])
				# 	print(latent[0])
				# 	print("")

				loss, BCE, KLD = loss_function(recon_batch, data_wo_acitonType, mu, logvar)
				loss.backward()
				train_loss_BCE += BCE.item() / 4096
				train_loss_KLD += KLD.item() / 4096


				# self.optimizers[0].step()
				self.optimizer.step()


			action_train_loss_BCE += train_loss_BCE / 4
			action_train_loss_KLD += train_loss_KLD / 4

		summary.add_scalars('actionType',{'BCE' : action_train_loss_BCE, 'KLD' : action_train_loss_KLD}, num_evaluation)


		self.CAAEDecoder.save(vaeDir + "/vae_action_decoder.pt")	
		self.CAAEEncoder.save(vaeDir + "/vae_action_encoder.pt")	



ccvt = ComprehensiveControlVectorTraining("basket_0", 5)
for i in range(1000):
	ccvt.trainTargetCV(i)

# ccvt = ComprehensiveControlVectorTraining("basket_0", 5)
# for i in range(400):
# 	ccvt.trainTargetCV(0, i)
# 	ccvt.trainTargetCV(1, i)
# 	ccvt.trainTargetCV(2, i)
# 	ccvt.trainTargetCV(3, i)
# 	ccvt.trainTargetCV(4, i)
