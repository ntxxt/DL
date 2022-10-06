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

nb_users = int(max(max(training_dataset[:,0]), max(test_dataset[:,0])))
nb_moives = int(max(max(training_dataset[:,1]), max(test_dataset[:,1])))

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

