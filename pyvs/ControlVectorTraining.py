import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils.dataLoader import loadData
from IPython import embed

import os
import sys
import random

from VAE import VAE, loss_function

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# print("asdf")
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
	model.save("vae_nn/vae_0.pt")	

def test(path):
	model.eval()
	with torch.no_grad():
		control_vector_list = loadData(os.path.dirname(os.path.abspath(__file__)) + "/../extern/ICA/motions/%s/data/0_xData.dat"%(path))
		random.shuffle(control_vector_list)
		model.load("vae_nn/vae_0.pt")
		for i in range(10):
			data = control_vector_list[i]
			data = Tensor(data)
			recon_batch , mu, logvar = model(data)
			print(data)
			print(recon_batch)
			print(mu)
			print(logvar)
			print("")



trainControlVector("basket_0")
# test("basket_0")