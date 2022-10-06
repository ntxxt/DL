#feature detection tool, recommdation system
#overcomplete hidden layers problem-more hidden nodes than input nodes
#solutions: sparse autoencoders, reguliaztion techinques applied, does not allow use all hidden nodes at same time
#Denoising autoencoders, replace input value with modified input values
#Contractive autoencoders, add penalty which does not allow copy and paste
#stacked autoencoders, encoding-encoding-decoding
#deepautoencoders, RBM stacked with autoencoding machenism

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

path = '/Users/xiaotongxu/Downloads/ml-1m'
movies = pd.read_csv(os.path.join(path, 'movies.dat'),sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv(os.path.join(path, 'users.dat'),sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv(os.path.join(path, 'ratings.dat'),sep='::', header=None, engine='python', encoding='latin-1')

training_dataset = pd.read_csv('/Users/xiaotongxu/Downloads/ml-100k/u1.base', delimiter='\t')
training_dataset = np.array(training_dataset, dtype='int')
test_dataset = pd.read_csv('/Users/xiaotongxu/Downloads/ml-100k/u1.test', delimiter='\t')
test_dataset = np.array(test_dataset, dtype='int')
print(test_dataset[0])
print(test_dataset.shape)

nb_users = int(max(max(training_dataset[:,0], ), max(test_dataset[:,0])))
nb_moives = int(max(max(training_dataset[:,1], ), max(test_dataset[:,1])))
print(nb_users, nb_moives)

def gen_matrix(dataset):
    new_data = []   
    for id_user in range(1, nb_users + 1):
        id_movies = dataset[:,1][dataset[:,0] == id_user]
        id_ratings = dataset[:,2][dataset[:,0] == id_user]
        ratings = np.zeros(nb_moives)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data
test_matrix = gen_matrix(test_dataset)
train_matrix = gen_matrix(training_dataset)

#matrix -> torch tensors
training_set = torch.FloatTensor(train_matrix)
test_set = torch.FloatTensor(test_matrix)
print(training_set[0])
print(training_set.shape)

class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_moives, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_moives)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)


nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = input.clone()
    if torch.sum(target.data > 0) > 0:
      output = sae(input)
      target.require_grad = False
      output[target == 0] = 0
      loss = criterion(output, target)
      mean_corrector = nb_moives/float(torch.sum(target.data > 0) + 1e-10)
      loss.backward()
      train_loss += np.sqrt(loss.data*mean_corrector)
      s += 1.
      optimizer.step()
  print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))

test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0 ] = 0
        loss = criterion(output, target)
        mean_corrector = nb_moives / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print(epoch, loss)


