from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython import embed
import numpy as np
# parser = argparse.ArgumentParser(description='VAE MNIST Example')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed)

# device = torch.device("cuda" if args.cuda else "cpu")

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
cvSize = 16
latentSize = 4

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.cvSize = 14
        scale = 256
        self.fc1 = nn.Linear( self.cvSize , scale*2)
        self.fc1b = nn.Linear(scale*2, scale)
        self.fc21 = nn.Linear(scale, latentSize)
        self.fc2122 = nn.Linear(scale, latentSize)
        self.fc3 = nn.Linear(latentSize, scale)
        self.fc3b = nn.Linear(scale, scale*2)
        self.fc4 = nn.Linear(scale*2,  self.cvSize )

        # self.fc1 = nn.Linear(14, 512)
        # self.fc21 = nn.Linear(512, 4)
        # self.fc22 = nn.Linear(512, 4)
        # self.fc3 = nn.Linear(4, 512)
        # self.fc4 = nn.Linear(512, 14)



        # self.fc1 = nn.Linear(784, 400)
        # self.fc21 = nn.Linear(400, 20)
        # self.fc22 = nn.Linear(400, 20)
        # self.fc3 = nn.Linear(20, 400)
        # self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1b = F.relu(self.fc1b(h1))
        return self.fc21(h1b), self.fc22(h1b)



    def decode(self, z):
        # embed()
        # exit(0)

        h3 = F.relu(self.fc3(z))
        h3b = F.relu(self.fc3b(h3))
        return self.fc4(h3b)

    def decodeAction(self, z):
        z = torch.tensor(z)
        z = z.cuda()
        h3 = F.relu(self.fc3(z))
        h3b = F.relu(self.fc3b(h3))
        # embed()
        # exit(0)
        return self.fc4(h3b).cpu().detach().numpy().astype(np.float32)


    # def encode(self, x):
    #     h1 = F.relu(self.fc1(x))
    #     return self.fc21(h1), self.fc22(h1)

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps*std

    # def decode(self, z):
    #     # embed()
    #     # exit(0)

    #     h3 = F.relu(self.fc3(z))
    #     return self.fc4(h3)

    # def decodeAction(self, z):
    #     z = torch.tensor(z)
    #     z = z.cuda()
    #     h3 = F.relu(self.fc3(z))
    #     # embed()
    #     # exit(0)
    #     return self.fc4(h3).cpu().detach().numpy().astype(np.float32)



    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.cvSize))
        # mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def load(self,path):
        print('load nn {}'.format(path))
        self.load_state_dict(torch.load(path))

    def save(self,path):
        print('save nn {}'.format(path))
        torch.save(self.state_dict(),path)


class VAEEncoder(nn.Module):
    def __init__(self):
        super(VAEEncoder, self).__init__()

        scale = 256
        self.fc1 = nn.Linear(cvSize+5, scale)
        self.fc1a = nn.Linear(scale, scale)
        self.fc1b = nn.Linear(scale, scale)
        self.fc21 = nn.Linear(scale, latentSize)
        self.fc22 = nn.Linear(scale, latentSize)


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1a = F.relu(self.fc1a(h1))
        h1b = F.relu(self.fc1b(h1a))
        return self.fc21(h1b), self.fc22(h1b)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, cvSize+5))
        # mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


    def load(self,path):
        print('load nn {}'.format(path))
        self.load_state_dict(torch.load(path))

    def save(self,path,verbose=False):
        if verbose:
            print('save nn {}'.format(path))
        torch.save(self.state_dict(),path)

class VAEDecoder(nn.Module):
    def __init__(self):
        super(VAEDecoder, self).__init__()

        scale = 256
        self.fc3 = nn.Linear(latentSize, scale)
        self.fc3a = nn.Linear(scale, scale)
        self.fc3b = nn.Linear(scale, scale)
        self.fc4 = nn.Linear(scale, cvSize)


    def decode(self, z):
        # embed()
        # exit(0)

        h3 = F.relu(self.fc3(z))
        h3a = F.relu(self.fc3a(h3))
        h3b = F.relu(self.fc3b(h3a))
        return self.fc4(h3b)

    def decodeAction(self, z):
        z = torch.tensor(z)
        z = z.cuda()
        h3 = F.relu(self.fc3(z))
        h3a = F.relu(self.fc3a(h3))
        h3b = F.relu(self.fc3b(h3a))
        # embed()
        # exit(0)
        return self.fc4(h3b).cpu().detach().numpy().astype(np.float32)


    def forward(self, x):
        # mu, logvar = self.encode(x.view(-1, 14))
        # # mu, logvar = self.encode(x.view(-1, 784))
        # z = self.reparameterize(mu, logvar)
        return self.decode(x)

    def load(self,path):
        print('load nn {}'.format(path))
        self.load_state_dict(torch.load(path))

    def save(self,path,verbose=False):
        if verbose:
            print('save nn {}'.format(path))
        torch.save(self.state_dict(),path)
# model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, cvSize), reduction='sum')
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # print(BCE.item(), end=", ")
    # print(KLD.item())
    # print("")

    return BCE + 0.5*KLD, BCE, 0.5*KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # embed()
        # exit(0)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')