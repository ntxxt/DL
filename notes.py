from operator import le
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

#tensors
scalar = torch.tensor(7)
vector = torch.tensor([7,7], dtype=float,device='cpu', requires_grad=False) #device; track gradients or not 
MATRIX = torch.tensor([[2,2],[3,3]])
TENSOR = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
random_tensor = torch.rand(3,4)
image = torch.rand(size=(3,224,224)) # (color channel, height, width)
zeros = torch.zeros(3,4)
a1 = torch.arange(1,10)
torch.matmul(TENSOR, TENSOR) #dot product; inner dimensions must match; have the shape of outer dimensions 
t1 = torch.rand(2,3)
t2 = torch.rand(2,3)
torch.matmul(t1.T, t2) #use transpose to fix shape issue
t1.argmin() #index for min 
a1_reshaped = a1.reshape(3,3) #i * j must equal a1.shape
a1_view = a1.view(3,3) #a1_view share memory with a1
a1_stack = torch.stack([a1, a1, a1], dim=1) #stack 
a1_squeeze = a1_stack.squeeze() #remove all single dimensions
x = torch.rand(2,3,2)
a1_permute  = torch.permute(x, (1,0,2)).size()