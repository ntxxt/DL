#Boltzmann Machines, deep belief networks, deep Boltzmann Machines
#no directions, no output layer, 
#whole system together, feed input data, help to adjust the weight
#so it actually  resemble our system
#Restricted Boltzmann Machines
#contrastive divergence
##adjust the energy curve by modifying the weights to facilitate a system, which resembles our input values

#
from tkinter import W
import pandas as pd
import torch 
import os
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

path = '/Users/xiaotongxu/Downloads/ml-1m'
movies = pd.read_csv(os.path.join(path, 'movies.dat'),sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv(os.path.join(path, 'users.dat'),sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv(os.path.join(path, 'ratings.dat'),sep='::', header=None, engine='python', encoding='latin-1')
'''
ratings data
viewer id; movie id; rating given by viewer; time stemp
'''
#print(ratings.head())
#print(movies.head())

training_dataset = pd.read_csv('/Users/xiaotongxu/Downloads/ml-100k/u1.base', delimiter='\t')
#dataframe to array
training_dataset = np.array(training_dataset, dtype='int')
test_dataset = pd.read_csv('/Users/xiaotongxu/Downloads/ml-100k/u1.test', delimiter='\t')
test_dataset = np.array(test_dataset, dtype='int')

nb_users = int(max(max(training_dataset[:,0]), max(test_dataset[:,0])))
nb_moives = int(max(max(training_dataset[:,1]), max(test_dataset[:,1])))
#print(nb_users)

#arrary -> matrix
#observations in lines, features in columns
#list of list, 942 users(lists), each users give rating to movie, if not, 0
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
training_tensor = torch.FloatTensor(train_matrix)
test_tensor = torch.FloatTensor(test_matrix)

#turn into binary ratings
training_tensor[training_tensor ==0] = -1
training_tensor[training_tensor == 1] = 0
training_tensor[training_tensor == 2] = 0
training_tensor[training_tensor >= 3] = 1

test_tensor[test_tensor ==0] = -1
test_tensor[test_tensor == 1] = 0
test_tensor[test_tensor == 2] = 0
test_tensor[test_tensor >= 3] = 1

#RBM wit pytorch
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.T, phk)).t()
        self.b += torch.sum((v0 -vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

RBM = RBM(nv=len(training_tensor[0]), nh=100)
batch_size = 100

#training

nb_epoch = 10
for i in range(1, nb_epoch+1):
    train_loss = 0
    counter = 0.
    for id_user in range(0, nb_users-batch_size, batch_size):
        vk = training_tensor[id_user:id_user+batch_size]
        v0 = training_tensor[id_user:id_user+batch_size]
        ph0, _ = RBM.sample_h(v0)
        for k in range(10):
            _, hk = RBM.sample_h(vk)
            _, vk = RBM.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk, _ = RBM.sample_h(vk)
        RBM.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        counter += 1.
    #print(i)


#test

test_loss = 0
counter = 0.
for id_user in range(nb_users):
    v = training_tensor[id_user:id_user+1]
    vt = test_tensor[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h= RBM.sample_h(v)
        _,v= RBM.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        counter += 1.
print(test_loss/counter)
