#feature detection tool, recommdation system
#overcomplete hidden layers problem-more hidden nodes than input nodes
#solutions: sparse autoencoders, reguliaztion techinques applied, does not allow use all hidden nodes at same time
#Denoising autoencoders, replace input value with modified input values
#Contractive autoencoders, add penalty which does not allow copy and paste
#stacked autoencoders, encoding-encoding-decoding
#deepautoencoders, RBM stacked with autoencoding machenism

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


